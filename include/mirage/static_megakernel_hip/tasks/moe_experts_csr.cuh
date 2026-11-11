// tasks/moe_experts_csr.cuh (HIP, gfx950) -- moe_experts for T > 16 tokens (plain decode at batch sizes 24 .. 64): the same
// work queues, W13 / W2 items, MFMAs and output as moe_experts.cuh (read that first), with what grows with T kept out of LDS
// (z_q alone is T x LAT bytes, 229 KB at T = 64):
//   - the routing table in CSR form: per expert a 64-bit token mask (an OR: every CU builds the same table); slots = the
//     experts used, in ascending id; slot s's pairs at [start[s], start[s + 1]) of the pair lists (token, k, weight),
//     tokens ascending
//   - z_q is not copied to LDS whole: a W13 unit is a CU's (a slot, 4 items: one per wave), the CU stages the slot's
//     tokens' z_q rows one K quarter (7 steps) at a time, all its tokens (at most 64), in the MFMA's fragment order
//     (system-scope loads, re-loaded until no byte is 0xFF: sum_quant_send's bytes are their own flags, as
//     moe_experts.cuh's W2 h_q); each wave's weights are read once for all the slot's tokens, 16 per MFMA (MI355X dec23:
//     4 tokens staged per wave and a unit run once per 4 tokens re-read the weights: T = 64 with 66 experts 594 us)
//   - h_q by pair: [NPAIR][IR / 128][4 lane groups][32 B], scales [NPAIR][IR / 128][4] (0xFF before each launch; W2 polls
//     them as moe_experts.cuh)
//   - a W2 item takes its slot's tokens 16 at a time (the weights loaded once)
#pragma once
#include "moe_experts.cuh"

namespace static_mk {

// STATIC_CSR_W2_TILES: row tiles per W2 item (8)
#ifndef STATIC_CSR_W2_TILES
#define STATIC_CSR_W2_TILES 8
#endif

template <int NE_,
          int K_,
          int LAT_,
          int IR_,
          int CNT_,
          size_t ZOFF_,
          int W13M_ = -1,
          int W13SFM_ = -1,
          int W2M_ = -1,
          int W2SFM_ = -1>
struct ExpertCfgC {
  static constexpr int NE = NE_, K = K_, LAT = LAT_, IR = IR_;
  static constexpr int W13M = W13M_, W13SFM = W13SFM_, W2M = W2M_, W2SFM = W2SFM_;
  static constexpr int NPAIR = T * K, NSLOT = NE < NPAIR ? NE : NPAIR;
  static constexpr int CNT = CNT_;
  static constexpr size_t ZOFF = ZOFF_;
  static constexpr int KS1 = LAT / 128, KG1 = KS1 / 4, RT1 = 2 * IR / 16, ITEMS1 = IR / 32;
  static constexpr size_t MISC_TRACE = 16 * 128 + 8192;
  static constexpr int KS2 = IR / 128, RT2 = LAT / 16, W2_TILES = STATIC_CSR_W2_TILES, ITEMS2 = RT2 / W2_TILES;
  static constexpr int STAGES = 7;
  static constexpr int SCW = STATIC_MOE_AITER ? 4 : KS2; // (moe_experts.cuh STATIC_MOE_AITER)
  static constexpr int KQ = KS1 / 4, G1 = ITEMS1 / 4; // a K quarter's steps; W13 units (4 items) per slot
  static_assert(LAT % 512 == 0 && IR % 128 == 0 && IR / 128 <= 4, "moe_experts_csr: as moe_experts");
  static_assert(RT2 % W2_TILES == 0 && W2_TILES % 4 == 0 && KS1 % STAGES == 0 && KS1 > STAGES,
                "moe_experts_csr: as moe_experts");
  static_assert(KS1 % 4 == 0 && ITEMS1 % 4 == 0, "moe_experts_csr: K quarters, units of 4 items");
  static_assert(NE % 64 == 0 && NE / 64 <= 16 && T <= 64, "moe_experts_csr: 64-expert blocks, a 64-bit token mask");
};

template <class C>
struct ExpertLdsC {
  static constexpr int ZC = 0;                                             // [KQ][4 g][64 tokens][32 B] fp8 (the CU's)
  static constexpr int ZCS = ZC + C::KQ * 4 * 64 * 32;                     // [KQ][4 g][64 tokens] e8m0
  static constexpr int WS = (ZCS + C::KQ * 4 * 64 + 15) / 16 * 16;         // [4 waves][4][KG1][64] u32
  static constexpr int MLO = WS + 4 * 4 * C::KG1 * 64 * 4;                 // [NE] u32: tokens 0 .. 31 that picked e
  static constexpr int MHI = MLO + C::NE * 4;                              // [NE] u32: tokens 32 .. 63
  static constexpr int E2S = MHI + C::NE * 4;                              // [NE] int: e's slot
  static constexpr int EID = E2S + C::NE * 4;                              // [NSLOT] int
  static constexpr int START = EID + C::NSLOT * 4;                         // [NSLOT + 1] int
  static constexpr int SEL = START + (C::NSLOT + 1) * 4;                   // [NPAIR] int: pair i's expert
  static constexpr int WSEL = SEL + C::NPAIR * 4;                          // [NPAIR] float: pair i's weight
  static constexpr int PWT = WSEL + C::NPAIR * 4;                          // [NPAIR] float (CSR order)
  static constexpr int PTOK = PWT + C::NPAIR * 4;                          // [NPAIR] u8
  static constexpr int PK = PTOK + C::NPAIR;                               // [NPAIR] u8
  static constexpr int BLK = (PK + C::NPAIR + 15) / 16 * 16; // [16] live experts, [16] pairs per 64-expert block, n_live
  static constexpr int CUQ = BLK + 33 * 4;                                 // the CU's W13 unit
  static constexpr int BYTES = CUQ + 16;
};

// no byte of the 16 is 0xFF
__device__ __forceinline__ bool ff_free16(uint4 const &u) {
  auto f = [](uint32_t w) {
    uint32_t const t = ~w;
    return ((t - 0x01010101u) & ~t & 0x80808080u) == 0u;
  };
  return f(u.x) && f(u.y) && f(u.z) && f(u.w);
}

#ifndef STATIC_CSR_PAIRS_PAR
#define STATIC_CSR_PAIRS_PAR 0
#endif
static_assert(STATIC_CSR_W2_TILES == 8, "moe_experts_csr: 8 row tiles per W2 item");
template <class C>
__device__ __forceinline__ void expert_zq_cheap_wait(G const &g) {
  uint32_t const zoff = (uint32_t)(C::ZOFF + rg_set());
  if ((int)STATIC_TIDX < C::KS1) {
    uint32_t const at = zoff + (uint32_t)(zq_scales_at(C::LAT) + (size_t)STATIC_TIDX * ZQ_CHUNK + (T - 1) * 16);
    while (!ff_free16(make_uint4(ld16_at<AUX_SYS>(g.rv, at).x, 0u, 0u, 0u))) {
      __builtin_amdgcn_s_sleep(8);
      asm volatile("" ::: "memory");
    }
  }
  __syncthreads();
}
template <class C>
__device__ __forceinline__ int expert_table_csr(unsigned long long const *pairs, char *sm) {
  using Ls = ExpertLdsC<C>;
  unsigned *mlo = reinterpret_cast<unsigned *>(sm + Ls::MLO), *mhi = reinterpret_cast<unsigned *>(sm + Ls::MHI);
  int *e2s = reinterpret_cast<int *>(sm + Ls::E2S), *eid = reinterpret_cast<int *>(sm + Ls::EID),
      *start = reinterpret_cast<int *>(sm + Ls::START), *sel = reinterpret_cast<int *>(sm + Ls::SEL),
      *blk = reinterpret_cast<int *>(sm + Ls::BLK);
  float *wsel = reinterpret_cast<float *>(sm + Ls::WSEL), *pwt = reinterpret_cast<float *>(sm + Ls::PWT);
  unsigned char *ptok = reinterpret_cast<unsigned char *>(sm + Ls::PTOK), *pk = reinterpret_cast<unsigned char *>(sm + Ls::PK);
  int const tid = task_tid_all();
  for (int i = tid; i < C::NE; i += 256) {
    mlo[i] = 0u;
    mhi[i] = 0u;
  }
  __syncthreads();
#if STATIC_CSR_PAIRS_PAR
  { // (STATIC_CSR_PAIRS_PAR: a thread's NPAIR / 256 pairs polled with their loads in flight together; default: one pair's
    // poll after another, a device-scope round trip each, 4 in a row at T = 64)
    constexpr int PI = (C::NPAIR + 255) / 256;
    unsigned long long v[PI];
    bool ok[PI];
#pragma unroll
    for (int j = 0; j < PI; j++) {
      ok[j] = tid + 256 * j >= C::NPAIR;
    }
    bool all;
    do {
      all = true;
#pragma unroll
      for (int j = 0; j < PI; j++) {
        if (!ok[j]) {
          v[j] = ld64_dev(pairs + tid + 256 * j);
          ok[j] = v[j] != ~0ull;
        }
        all = all && ok[j];
      }
      if (!all) {
        __builtin_amdgcn_s_sleep(1);
      }
    } while (!all);
#pragma unroll
    for (int j = 0; j < PI; j++) {
      int const i = tid + 256 * j;
      if (i < C::NPAIR) {
        int const e = (int)(v[j] >> 32), t = i / C::K;
        sel[i] = e;
        wsel[i] = __uint_as_float((uint32_t)v[j]);
        atomicOr(t < 32 ? &mlo[e] : &mhi[e], 1u << (t & 31));
      }
    }
  }
#else
  for (int i = tid; i < C::NPAIR; i += 256) { // pair i (0xFF-prefilled, polled)
    unsigned long long v;
    while ((v = ld64_dev(pairs + i)) == ~0ull) {
      __builtin_amdgcn_s_sleep(1);
    }
    int const e = (int)(v >> 32), t = i / C::K;
    sel[i] = e;
    wsel[i] = __uint_as_float((uint32_t)v);
    atomicOr(t < 32 ? &mlo[e] : &mhi[e], 1u << (t & 31));
  }
#endif
  __syncthreads();
  // slots in ascending expert id: 64-expert blocks, wave w takes blocks w, w + 4, ..; per block its live experts (a
  // ballot) and pairs (a wave sum)
  int const wave = tid >> 6, lane = tid & 63;
  constexpr int NB = C::NE / 64;
  unsigned long long m[4];
  int np[4];
#pragma unroll
  for (int q = 0; q < 4; q++) {
    int const b = wave + 4 * q;
    int n = 0;
    if (b < NB) {
      n = __popc(mlo[b * 64 + lane]) + __popc(mhi[b * 64 + lane]);
    }
    np[q] = n;
    m[q] = __ballot(n > 0);
    int s = n;
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) {
      s += __shfl_xor(s, o);
    }
    if (b < NB && lane == 0) {
      blk[b] = __popcll(m[q]);
      blk[16 + b] = s;
    }
  }
  __syncthreads();
#pragma unroll
  for (int q = 0; q < 4; q++) {
    int const b = wave + 4 * q;
    if (b < NB) { // (wave-uniform)
      int sbase = 0, pbase = 0;
      for (int b_ = 0; b_ < b; b_++) {
        sbase += blk[b_];
        pbase += blk[16 + b_];
      }
      int x = np[q]; // the pairs of the block's experts up to this lane's
#pragma unroll
      for (int d = 1; d < 64; d <<= 1) {
        int const y = __shfl_up(x, d);
        if (lane >= d) {
          x += y;
        }
      }
      if (np[q] > 0) {
        int const e = b * 64 + lane, slot = sbase + __popcll(m[q] & ((1ull << lane) - 1ull));
        eid[slot] = e;
        e2s[e] = slot;
        start[slot] = pbase + x - np[q];
      }
    }
  }
  if (tid == 0) {
    int n = 0;
    for (int b_ = 0; b_ < NB; b_++) {
      n += blk[b_];
    }
    blk[32] = n;
    start[n] = C::NPAIR; // (each token's K experts are distinct: every pair in one slot)
  }
  __syncthreads();
  for (int i = tid; i < C::NPAIR; i += 256) { // pair -> its CSR position (tokens ascending within a slot)
    int const e = sel[i], t = i / C::K;
    unsigned const lo = mlo[e], hi = mhi[e];
    int const p = t < 32 ? __popc(lo & ((1u << t) - 1u)) : __popc(lo) + __popc(hi & ((1u << (t - 32)) - 1u));
    int const idx = start[e2s[e]] + p;
    ptok[idx] = (unsigned char)t;
    pk[idx] = (unsigned char)(i % C::K);
    pwt[idx] = wsel[i];
  }
  __syncthreads();
  return blk[32];
}

// STATIC_CSR_ZRND (default 7): z_q pieces per thread per polling round of a W13 unit's staging - a quarter of 64 tokens is 14
// per thread: 7 = two rounds per quarter, eight round trips per unit (MI355X dec221, T 64: the many-token slots' units 36-56 us
// of which 14-25 staging, at the experts' end)
#ifndef STATIC_CSR_ZRND
#define STATIC_CSR_ZRND 7
#endif
// STATIC_CSR_ZDMA 1 (default 0; with sum_quant_send's STATIC_SQS_SCALES_LAST): before a CU's first W13 unit every z_q scale row
// (KS1 x T 16-B rows) is polled once until landed - then every block's bytes have landed - and the units' z_q quarters are
// staged by LDS DMA, all of a quarter's pieces in flight, no registers held, no re-polls (MI355X dec221 / dec222, T 64: the
// many-token slots' units spent 14-25 us in the polled staging, more pieces per round cost registers in the fused kernel)
#ifndef STATIC_CSR_ZDMA
#define STATIC_CSR_ZDMA 0
#endif
template <class C>
__device__ __forceinline__ void expert_zq_all_wait(G const &g) {
  uint32_t const zoff = (uint32_t)(C::ZOFF + rg_set());
  for (int v = STATIC_TIDX; v < C::KS1 * T; v += 256) { // scale row (chunk v / T, token v % T)
    uint32_t const at = zoff + (uint32_t)(zq_scales_at(C::LAT) + (size_t)(v / T) * ZQ_CHUNK + (v % T) * 16);
    while (!ff_free16(make_uint4(ld16_at<AUX_SYS>(g.rv, at).x, 0u, 0u, 0u))) {
      __builtin_amdgcn_s_sleep(4);
      asm volatile("" ::: "memory");
    }
  }
  __syncthreads();
}
// K quarter q of the slot's tokens tk[0 .. n) by LDS DMA (every block landed: expert_zq_all_wait): piece v = ((ks 4 + gq) 64 + j)
// 2 + h at 16 v of the CU's area (the polled staging's layout), j >= n from token tk[0] (columns never stored); scales by loads
template <class C>
__device__ __forceinline__ void expert_stage_zq_dma(G const &g, char *sm, int q, unsigned char const *tk, int n) {
  using Ls = ExpertLdsC<C>;
  unsigned char *const zc = reinterpret_cast<unsigned char *>(sm + Ls::ZC);
  unsigned char *const zcs = reinterpret_cast<unsigned char *>(sm + Ls::ZCS);
  unsigned char const *const zb = g.rv + C::ZOFF + rg_set();
  int const lane = STATIC_TIDX & 63, wave = __builtin_amdgcn_readfirstlane((int)STATIC_TIDX >> 6);
#pragma unroll 2
  for (int base = 64 * wave; base < C::KQ * 512; base += 256) {
    int const v = base + lane, h = v & 1, j = (v >> 1) & 63, gq = (v >> 7) & 3, ks = v >> 9;
    int const tok = tk[j < n ? j : 0];
    __builtin_amdgcn_global_load_lds(
        (__attribute__((address_space(1))) void *)(const_cast<unsigned char *>(zb) + (size_t)tok * C::LAT +
                                                    128 * (C::KQ * q + ks) + 64 * h + 16 * gq),
        (__attribute__((address_space(3))) void *)(zc + (size_t)base * 16), 16, 0, AUX_SYS);
  }
  for (int v = STATIC_TIDX; v < C::KQ * n; v += 256) {
    int const j = v % n, ks = v / n;
    uint32_t const sv = ld16_at<AUX_SYS>(g.rv, (uint32_t)(C::ZOFF + rg_set() + zq_scales_at(C::LAT) +
                                                            (size_t)(C::KQ * q + ks) * ZQ_CHUNK + tk[j] * 16)).x;
#pragma unroll
    for (int gq = 0; gq < 4; gq++) {
      zcs[(ks * 4 + gq) * 64 + j] = (unsigned char)(sv >> (8 * gq));
    }
  }
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory"); // (the DMA's loads count in vmcnt only)
}
// a W13 unit's z_q, K quarter q: the rows of the slot's tokens tk[0 .. n) (n <= 64) into the CU's area, every thread,
// polled (the caller syncs before the reads)
template <class C>
__device__ __forceinline__ void expert_stage_zq(G const &g, char *sm, int q, unsigned char const *tk, int n) {
  using Ls = ExpertLdsC<C>;
  unsigned char *const zc = reinterpret_cast<unsigned char *>(sm + Ls::ZC);
  unsigned char *const zcs = reinterpret_cast<unsigned char *>(sm + Ls::ZCS);
  uint32_t const zoff = (uint32_t)(C::ZOFF + rg_set());
  // piece v = (ks, gq, h, j), j fastest: 16 B of token j's row at 128 (KQ q + ks) + 64 h + 16 gq -> ((ks 4 + gq) 64 + j) 32
  // + 16 h
  int const nzp = C::KQ * 8 * n;
  constexpr int RND = STATIC_CSR_ZRND; // pieces per thread in flight per round
#pragma unroll 1
  for (int v0 = 0; v0 < nzp; v0 += 256 * RND) {
    uint4 u[RND];
    bool ok[RND];
#pragma unroll
    for (int r = 0; r < RND; r++) {
      ok[r] = v0 + (int)STATIC_TIDX + 256 * r >= nzp;
      u[r] = make_uint4(0u, 0u, 0u, 0u);
    }
    bool all;
    do {
#pragma unroll
      for (int r = 0; r < RND; r++) {
        if (!ok[r]) {
          int const v = v0 + (int)STATIC_TIDX + 256 * r, j = v % n, rest = v / n, h = rest & 1, gq = (rest >> 1) & 3,
                    ks = rest >> 3;
          u[r] = ld16_at<AUX_SYS>(g.rv, zoff + (uint32_t)(tk[j] * C::LAT + 128 * (C::KQ * q + ks) + 64 * h + 16 * gq));
        }
      }
      all = true;
#pragma unroll
      for (int r = 0; r < RND; r++) {
        ok[r] = ok[r] || ff_free16(u[r]);
        all = all && ok[r];
      }
      if (!all) {
        poll_pause();
      }
    } while (!all);
#pragma unroll
    for (int r = 0; r < RND; r++) {
      int const v = v0 + (int)STATIC_TIDX + 256 * r;
      if (v < nzp) {
        int const j = v % n, rest = v / n, h = rest & 1, gq = (rest >> 1) & 3, ks = rest >> 3;
        *reinterpret_cast<uint4 *>(zc + ((ks * 4 + gq) * 64 + j) * 32 + 16 * h) = u[r];
      }
    }
  }
  // the scales: row (ks, t) of the scale chunks, bytes 0..3 = lane groups 0..3 (e8m0, never 0xFF)
  for (int v = STATIC_TIDX; v < C::KQ * n; v += 256) {
    int const j = v % n, ks = v / n;
    uint4 sv;
    do {
      sv = ld16_at<AUX_SYS>(g.rv, zoff + (uint32_t)(zq_scales_at(C::LAT) + (size_t)(C::KQ * q + ks) * ZQ_CHUNK + tk[j] * 16));
      if (ff_free16(make_uint4(sv.x, 0u, 0u, 0u))) {
        break;
      }
      poll_pause();
    } while (true);
#pragma unroll
    for (int gq = 0; gq < 4; gq++) {
      zcs[(ks * 4 + gq) * 64 + j] = (unsigned char)(sv.x >> (8 * gq));
    }
  }
}

// a wave's item it of slot e (4 row tiles): K quarter q's weights (KQ steps) into pf [KQ][4]; q == 0: the scales too,
// into this wave's WS area (read after the caller's sync)
template <class C>
__device__ __forceinline__ void expert_w13q_issue(Maps const &maps, char *sm, int e, int it, int q, v4i *pf) {
  using Ls = ExpertLdsC<C>;
  constexpr int NTW = 4, NG = C::KG1;
  static_assert(C::KQ * NTW <= W13_PF && NTW * NG <= W13_PS, "W13_PF / W13_PS");
  int const lane = STATIC_TIDX & 63, wave = STATIC_TIDX >> 6;
  v4i const *wt = reinterpret_cast<v4i const *>(maps.m[C::W13M].p) + ((size_t)e * C::RT1 + 4 * it) * C::KS1 * 64 + lane;
  if (q == 0) {
    unsigned const *st =
        reinterpret_cast<unsigned const *>(maps.m[C::W13SFM].p) + ((size_t)e * C::RT1 + 4 * it) * C::KG1 * 64 + lane;
    unsigned *ws_l = reinterpret_cast<unsigned *>(sm + Ls::WS) + wave * 4 * C::KG1 * 64;
    unsigned ps[NTW * NG];
#if STATIC_MOE_AITER
    unsigned raw[2 * NTW * NG];
    unsigned const *sa = reinterpret_cast<unsigned const *>(maps.m[C::W13SFM].p) + lane;
#pragma unroll
    for (int t = 0; t < NTW; t++)
#pragma unroll
      for (int g = 0; g < NG; g++) {
        unsigned const *p = moe_w13s_aiter<C>(sa, e, it, t, g);
        raw[2 * (t * NG + g)] = __builtin_nontemporal_load(p);
        raw[2 * (t * NG + g) + 1] = __builtin_nontemporal_load(p + 64);
      }
#pragma unroll
    for (int t = 0; t < NTW; t++)
#pragma unroll
      for (int g = 0; g < NG; g++) {
        ps[t * NG + g] = moe_w13s_pack(raw[2 * (t * NG + g)], raw[2 * (t * NG + g) + 1], moe_w13_tile(t) & 1);
      }
    (void)st;
#else
#pragma unroll
    for (int t = 0; t < NTW; t++)
#pragma unroll
      for (int g = 0; g < NG; g++) {
        ps[t * NG + g] = __builtin_nontemporal_load(st + ((size_t)t * C::KG1 + g) * 64);
      }
#endif
#pragma unroll
    for (int t = 0; t < NTW; t++)
#pragma unroll
      for (int g = 0; g < NG; g++) {
        ws_l[(t * C::KG1 + g) * 64 + lane] = ps[t * NG + g];
      }
  }
#pragma unroll
  for (int s = 0; s < C::KQ; s++)
#pragma unroll
    for (int t = 0; t < NTW; t++) {
#if STATIC_MOE_AITER
      pf[s * NTW + t] = wload(wt + ((size_t)moe_w13_tile(t) * C::KS1 + C::KQ * q + s) * 64);
#else
      pf[s * NTW + t] = wload(wt + ((size_t)t * C::KS1 + C::KQ * q + s) * 64);
#endif
    }
}
// K quarter q's MFMAs: the wave's 4 row tiles x the slot's token tiles (16 tokens each, 16 c < n; columns >= n zeros) from
// the CU's staged z_q
template <class C>
__device__ __forceinline__ void expert_w13q_run(char const *sm, int q, int n, v4i const *pf, v4f (*acc)[4]) {
  using Ls = ExpertLdsC<C>;
  constexpr int NTW = 4;
  int const lane = STATIC_TIDX & 63, wave = STATIC_TIDX >> 6;
  unsigned char const *const zc = reinterpret_cast<unsigned char const *>(sm + Ls::ZC);
  unsigned char const *const zcs = reinterpret_cast<unsigned char const *>(sm + Ls::ZCS);
  unsigned const *ws_l = reinterpret_cast<unsigned const *>(sm + Ls::WS) + wave * 4 * C::KG1 * 64;
  int const col = lane % 16, gq = lane / 16;
#pragma unroll
  for (int s = 0; s < C::KQ; s++) {
    int const ks = C::KQ * q + s;
#pragma unroll
    for (int c = 0; c < 4; c++) {
      if (16 * c < n) { // (wave-uniform)
        int const j = 16 * c + col;
        bool const live = j < n;
        v8i b = *reinterpret_cast<v8i const *>(zc + ((s * 4 + gq) * 64 + j) * 32);
        int const bs = live ? (int)zcs[(s * 4 + gq) * 64 + j] : 127;
        if (!live) {
          b = v8i{0, 0, 0, 0, 0, 0, 0, 0};
        }
#pragma unroll
        for (int t = 0; t < NTW; t++) {
          v4i const &f = pf[s * NTW + t];
          v8i const a = {f[0], f[1], f[2], f[3], 0, 0, 0, 0};
          int const as = (ws_l[(t * C::KG1 + ks / 4) * 64 + lane] >> (8 * (ks % 4))) & 0xFF;
          acc[c][t] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, acc[c][t], 4 /*A fp4*/, 0 /*B fp8*/, 0, as, 0,
                                                                       bs);
        }
      }
    }
  }
}

// A slot of at most 16 tokens: its z_q for all KS1 steps at once, [KS1][4 g][16 tokens]
// [32 B] in the area of a quarter of 64 tokens (the same bytes), the pieces and the scale rows polled in one loop (MI355X
// dec54, T = 64: a unit of 4 staged quarters 25.5 us, 9.1 of them staging, the scale rows one round trip each)
#ifndef STATIC_CSR_Z16_RND // the staging's pieces per thread per round (14: one round for 16 tokens)
#define STATIC_CSR_Z16_RND 7
#endif
template <class C>
__device__ __forceinline__ void expert_stage_z16(G const &g, char *sm, unsigned char const *tk, int n) {
  using Ls = ExpertLdsC<C>;
  static_assert(C::KS1 * 4 * 16 == C::KQ * 4 * 64, "expert_stage_z16: all K for 16 tokens in a quarter's area");
  unsigned char *const zc = reinterpret_cast<unsigned char *>(sm + Ls::ZC);
  unsigned char *const zcs = reinterpret_cast<unsigned char *>(sm + Ls::ZCS);
  uint32_t const zoff = (uint32_t)(C::ZOFF + rg_set());
  // piece v = (ks, gq, h, j), j fastest (n of them): 16 B of token j's row at 128 ks + 64 h + 16 gq; scale row v = (ks, j);
  // RND pieces per thread per round (a round's loads, the first round's with the scale rows, polled until all landed)
  int const nzp = C::KS1 * 8 * n, nsc = C::KS1 * n;
  constexpr int RND = STATIC_CSR_Z16_RND, PS = (C::KS1 * 16 + 255) / 256;
#pragma unroll 1
  for (int v0 = 0; v0 < nzp; v0 += 256 * RND) {
    uint4 u[RND];
    uint32_t us[PS];
    bool ok[RND], oks[PS];
#pragma unroll
    for (int r = 0; r < RND; r++) {
      ok[r] = v0 + (int)STATIC_TIDX + 256 * r >= nzp;
      u[r] = make_uint4(0u, 0u, 0u, 0u);
    }
#pragma unroll
    for (int r = 0; r < PS; r++) {
      oks[r] = v0 > 0 || (int)STATIC_TIDX + 256 * r >= nsc;
      us[r] = 0u;
    }
    bool all;
    do {
#pragma unroll
      for (int r = 0; r < RND; r++) {
        if (!ok[r]) {
          int const v = v0 + (int)STATIC_TIDX + 256 * r, j = v % n, rest = v / n, h = rest & 1, gq = (rest >> 1) & 3,
                    ks = rest >> 3;
          u[r] = ld16_at<AUX_SYS>(g.rv, zoff + (uint32_t)(tk[j] * C::LAT + 128 * ks + 64 * h + 16 * gq));
        }
      }
#pragma unroll
      for (int r = 0; r < PS; r++) {
        if (!oks[r]) {
          int const v = (int)STATIC_TIDX + 256 * r, j = v % n, ks = v / n;
          us[r] = ld16_at<AUX_SYS>(g.rv, zoff + (uint32_t)(zq_scales_at(C::LAT) + (size_t)ks * ZQ_CHUNK + tk[j] * 16)).x;
        }
      }
      all = true;
#pragma unroll
      for (int r = 0; r < RND; r++) {
        ok[r] = ok[r] || ff_free16(u[r]);
        all = all && ok[r];
      }
#pragma unroll
      for (int r = 0; r < PS; r++) {
        oks[r] = oks[r] || ff_free16(make_uint4(us[r], 0u, 0u, 0u));
        all = all && oks[r];
      }
      if (!all) {
        poll_pause();
      }
    } while (!all);
#pragma unroll
    for (int r = 0; r < RND; r++) {
      int const v = v0 + (int)STATIC_TIDX + 256 * r;
      if (v < nzp) {
        int const j = v % n, rest = v / n, h = rest & 1, gq = (rest >> 1) & 3, ks = rest >> 3;
        *reinterpret_cast<uint4 *>(zc + ((ks * 4 + gq) * 16 + j) * 32 + 16 * h) = u[r];
      }
    }
    if (v0 == 0) {
#pragma unroll
      for (int r = 0; r < PS; r++) {
        int const v = (int)STATIC_TIDX + 256 * r;
        if (v < nsc) {
          int const j = v % n, ks = v / n;
#pragma unroll
          for (int gq = 0; gq < 4; gq++) {
            zcs[(ks * 4 + gq) * 16 + j] = (unsigned char)(us[r] >> (8 * gq));
          }
        }
      }
    }
  }
}
// the small slot's MFMAs: the wave's 4 row tiles over all KS1 steps, one token tile, the weights KQ steps ahead (pf: steps
// 0 .. KQ - 1 from expert_w13q_issue(q = 0), the scales in WS)
template <class C>
__device__ __forceinline__ void expert_w13s_run(Maps const &maps, char const *sm, int e, int it, int n, v4i *pf, v4f *acc) {
  using Ls = ExpertLdsC<C>;
  constexpr int NTW = 4, STG = C::KQ;
  static_assert(C::KS1 % STG == 0 && C::KS1 > STG, "expert_w13s_run: whole rounds");
  int const lane = STATIC_TIDX & 63, wave = STATIC_TIDX >> 6, col = lane % 16, gq = lane / 16;
  unsigned char const *const zc = reinterpret_cast<unsigned char const *>(sm + Ls::ZC);
  unsigned char const *const zcs = reinterpret_cast<unsigned char const *>(sm + Ls::ZCS);
  unsigned const *ws_l = reinterpret_cast<unsigned const *>(sm + Ls::WS) + wave * 4 * C::KG1 * 64;
  v4i const *wt = reinterpret_cast<v4i const *>(maps.m[C::W13M].p) + ((size_t)e * C::RT1 + 4 * it) * C::KS1 * 64 + lane;
  bool const live = col < n;
#pragma unroll
  for (int t = 0; t < NTW; t++) {
    acc[t] = v4f{0.f, 0.f, 0.f, 0.f};
  }
  auto step = [&](int ks, v4i const *f) {
    v8i b = *reinterpret_cast<v8i const *>(zc + ((ks * 4 + gq) * 16 + col) * 32);
    int const bs = live ? (int)zcs[(ks * 4 + gq) * 16 + col] : 127;
    if (!live) {
      b = v8i{0, 0, 0, 0, 0, 0, 0, 0};
    }
#pragma unroll
    for (int t = 0; t < NTW; t++) {
      v8i const a = {f[t][0], f[t][1], f[t][2], f[t][3], 0, 0, 0, 0};
      int const as = (ws_l[(t * C::KG1 + ks / 4) * 64 + lane] >> (8 * (ks % 4))) & 0xFF;
      acc[t] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, acc[t], 4 /*A fp4*/, 0 /*B fp8*/, 0, as, 0, bs);
    }
  };
#pragma unroll 1
  for (int k0 = 0; k0 < C::KS1 - STG; k0 += STG) {
#pragma unroll
    for (int s = 0; s < STG; s++) {
      step(k0 + s, pf + s * NTW);
#pragma unroll
      for (int t = 0; t < NTW; t++) {
#if STATIC_MOE_AITER
        pf[s * NTW + t] = wload(wt + ((size_t)moe_w13_tile(t) * C::KS1 + k0 + s + STG) * 64);
#else
        pf[s * NTW + t] = wload(wt + ((size_t)t * C::KS1 + k0 + s + STG) * 64);
#endif
      }
    }
  }
#pragma unroll
  for (int s = 0; s < STG; s++) {
    step(C::KS1 - STG + s, pf + s * NTW);
  }
}

// SiTU, MXFP8 h_q of the unit's tokens -> h_q by pair (pairs idx0 .. idx0 + nz - 1)
template <class C>
__device__ __forceinline__ void expert_w13c_finish(int idx0,
                                                   int nz,
                                                   int it,
                                                   v4f const *gate,
                                                   v4f const *up,
                                                   unsigned char *hq,
                                                   unsigned char *hsc,
                                                   bool local = false) {
  int const lane = STATIC_TIDX & 63, tk = lane % 16;
  float hv[2][4], amax = 0.f;
#pragma unroll
  for (int half = 0; half < 2; half++)
#pragma unroll
    for (int r = 0; r < 4; r++) {
      hv[half][r] = situ(gate[half][r], up[half][r]);
      amax = fmaxf(amax, fabsf(hv[half][r]));
    }
  amax = fmaxf(amax, __shfl_xor(amax, 16));
  amax = fmaxf(amax, __shfl_xor(amax, 32));
  int ex = ceil_log2_bits(fmaxf(amax, 1.0e-30f) * (1.0f / 448.0f));
  ex = ex < -127 ? -127 : (ex > 127 ? 127 : ex);
  float const inv = __uint_as_float((uint32_t)(127 - ex) << 23);
  if (tk < nz) {
    int const idx = idx0 + tk, ks2 = it / 4, q = it % 4;
#pragma unroll
    for (int half = 0; half < 2; half++)
#pragma unroll
      for (int r = 0; r < 4; r++) {
        int const f = 16 * half + 4 * (lane / 16) + r;
        int gq, b;
        fp8_slot(32 * q + f, gq, b);
        if (local) { // (STATIC_CSR_HQ_LOCAL: a write-back store, kept in this XCD's L2 - every reader of the slot runs on this XCD)
          hq[(((size_t)idx * C::KS2 + ks2) * 4 + gq) * 32 + b] = cvt_e4m3(hv[half][r] * inv);
        } else {
          st8_out(hq + (((size_t)idx * C::KS2 + ks2) * 4 + gq) * 32 + b, cvt_e4m3(hv[half][r] * inv));
        }
      }
    if (lane < 16) {
      st8_out(hsc + ((size_t)idx * C::KS2 + ks2) * 4 + q, (uint8_t)(ex + 127));
    }
  }
}

// W2: an item's operands for the pairs idx0 .. idx0 + nc - 1 (nc <= 16)
template <class C>
struct W2ItemC {
  v8i b[C::KS2];
  int bs[C::KS2];
  v4i f[C::W2_TILES][C::KS2];
  unsigned sc[C::W2_TILES / 4][C::SCW];
#ifdef STATIC_TIMING_BUILD
  long long t_load = 0; // the timing build: when its loads were issued
#endif
};
template <class C>
__device__ __forceinline__ void expert_w2c_hq(W2ItemC<C> &F, int idx0, int nc, unsigned char const *hq,
                                              unsigned char const *hsc) {
  int const lane = STATIC_TIDX & 63, col = lane % 16, gq = lane / 16;
#pragma unroll
  for (int k = 0; k < C::KS2; k++) {
    F.b[k] = v8i{0, 0, 0, 0, 0, 0, 0, 0};
    F.bs[k] = 0;
    if (col < nc) {
      size_t const o = ((size_t)(idx0 + col) * C::KS2 + k) * 4 + gq;
      uint4 const lo = ld16_in_at(hq, (uint32_t)(o * 32)), hi = ld16_in_at(hq, (uint32_t)(o * 32 + 16));
      F.b[k] = v8i{(int)lo.x, (int)lo.y, (int)lo.z, (int)lo.w, (int)hi.x, (int)hi.y, (int)hi.z, (int)hi.w};
      F.bs[k] = ld8_in_at(hsc, (uint32_t)o);
    }
  }
}
template <class C>
__device__ __forceinline__ bool expert_w2c_ok(W2ItemC<C> const &F, int nc) {
  int const lane = STATIC_TIDX & 63;
  auto ff_free = [](uint32_t w) {
    uint32_t const t = ~w;
    return ((t - 0x01010101u) & ~t & 0x80808080u) == 0u;
  };
  bool ok = true;
  if (lane % 16 < nc) {
#pragma unroll
    for (int k = 0; k < C::KS2; k++) {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        ok = ok && ff_free((uint32_t)F.b[k][i]);
      }
      ok = ok && F.bs[k] != 0xFF;
    }
  }
  return __all(ok);
}
#ifndef STATIC_CSR_W2_NOSTEAL
#define STATIC_CSR_W2_NOSTEAL 0
#endif
#ifndef STATIC_CSR_HQ_LOCAL
#define STATIC_CSR_HQ_LOCAL 0
#endif
// STATIC_CSR_W2_HQ_FIRST 1 (default 0): an item's h_q / scale loads issued before its weights, so waiting for them (the ready
// check, the MFMAs' operands) never covers the item's or the next item's weight loads (MI355X 10-26 dlib54 T 64 lib.s: vmcnt(31)
// then vmcnt(0) right before each item's first MFMA - the next item's 24 KB drained once per item)
#ifndef STATIC_CSR_W2_HQ_FIRST
#define STATIC_CSR_W2_HQ_FIRST 0
#endif
template <class C>
__device__ __forceinline__ void expert_w2c_load(W2ItemC<C> &F, Maps const &maps, int e, int t0, int idx0, int nc,
                                                unsigned char const *hq, unsigned char const *hsc) {
  int const lane = STATIC_TIDX & 63;
#if STATIC_CSR_W2_HQ_FIRST
  expert_w2c_hq<C>(F, idx0, nc, hq, hsc);
#endif
  v4i const *wt = reinterpret_cast<v4i const *>(maps.m[C::W2M].p) + ((size_t)e * C::RT2 + t0) * C::KS2 * 64 + lane;
#if STATIC_MOE_AITER
  unsigned const *st = reinterpret_cast<unsigned const *>(maps.m[C::W2SFM].p) +
                       ((size_t)e * (C::RT2 / 2) + t0 / 2) * 2 * 64 + lane; // ([NE][LAT / 32][2][64])
#pragma unroll
  for (int q = 0; q < C::W2_TILES / 4; q++)
#pragma unroll
    for (int d = 0; d < 4; d++) {
      F.sc[q][d] = st[((size_t)(2 * q + d / 2) * 2 + d % 2) * 64];
    }
#else
  unsigned const *st =
      reinterpret_cast<unsigned const *>(maps.m[C::W2SFM].p) + (((size_t)e * C::RT2 + t0) / 4 * 64 + lane) * C::SCW;
#pragma unroll
  for (int q = 0; q < C::W2_TILES / 4; q++)
#pragma unroll
    for (int d = 0; d < C::SCW; d++) {
      F.sc[q][d] = st[(size_t)q * 64 * C::SCW + d];
    }
#endif
#pragma unroll
  for (int t = 0; t < C::W2_TILES; t++)
#pragma unroll
    for (int k = 0; k < C::KS2; k++) {
      F.f[t][k] = wload(wt + ((size_t)t * C::KS2 + k) * 64);
    }
#if !STATIC_CSR_W2_HQ_FIRST
  expert_w2c_hq<C>(F, idx0, nc, hq, hsc);
#endif
}
template <class C>
__device__ __forceinline__ void expert_w2c_ready(W2ItemC<C> &F, int idx0, int nc, unsigned char const *hq,
                                                 unsigned char const *hsc) {
  if (expert_w2c_ok<C>(F, nc)) {
    return;
  }
  do {
    w2_poll_pause();
    expert_w2c_hq<C>(F, idx0, nc, hq, hsc);
  } while (!expert_w2c_ok<C>(F, nc));
  __builtin_amdgcn_s_waitcnt(0); // (as expert_w2_ready)
}
struct W2NoHook {
  __device__ void operator()() const {}
};
// rows (t, k) of the pairs idx0 .. idx0 + nc - 1 = weight x result
template <class C, class HOOK = W2NoHook>
__device__ __forceinline__ void expert_w2c_compute(W2ItemC<C> &F, char const *sm, int idx0, int nc, int t0, float *out,
                                                   HOOK &&hook = HOOK{}) {
  using Ls = ExpertLdsC<C>;
  int const lane = STATIC_TIDX & 63, p = lane % 16;
  int t = 0, k = 0;
  float w = 0.f;
  if (p < nc) {
    t = reinterpret_cast<unsigned char const *>(sm + Ls::PTOK)[idx0 + p];
    k = reinterpret_cast<unsigned char const *>(sm + Ls::PK)[idx0 + p];
    w = reinterpret_cast<float const *>(sm + Ls::PWT)[idx0 + p];
  }
  float *row_out = out + ((size_t)t * C::K + k) * C::LAT;
#pragma unroll
  for (int i = 0; i < C::W2_TILES; i++) {
    v4f acc = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int kq = 0; kq < C::KS2; kq++) {
      v8i const a = {F.f[i][kq][0], F.f[i][kq][1], F.f[i][kq][2], F.f[i][kq][3], 0, 0, 0, 0};
      acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
          a, F.b[kq], acc, 4, 0, 0,
          moe_w2_sbyte<C>(F.sc[i / 4], i % 4, kq), 0, F.bs[kq]);
    }
    if (p < nc) {
      int const row = (t0 + i) * 16 + 4 * (lane / 16);
      st16_out(row_out + row, make_uint4(__float_as_uint(w * acc[0]), __float_as_uint(w * acc[1]),
                                         __float_as_uint(w * acc[2]), __float_as_uint(w * acc[3])));
    }
  }
}

// the task of one CU (as moe_experts_task: the queues, the W2 counts per 128-column block of R)
template <class C>
__device__ __forceinline__ void moe_experts_csr_task(Maps const &maps,
                                                     G const &g,
                                                     KernelLocals &L,
                                                     unsigned long long const *pairs,
                                                     unsigned char *hq,
                                                     unsigned char *hsc,
                                                     uint32_t *qs,
                                                     unsigned long long *trace,
                                                     float *out) {
  using Ls = ExpertLdsC<C>;
  char *const sm = L.sm;
  __syncthreads(); // the LDS below may still be read by the previous task
#ifdef STATIC_TIMING_BUILD
  // the item trace (as moe_experts_task's: per wave 32 entries of 2 u64, start ns, duration ns | kind << 32 | slot << 40):
  // kind 10 task start -> table, 1 a W13 unit, 14 its z_q staging (the 4 quarters' stage + sync, summed, from the unit's
  // start), 2 a W2 item (its loads issued -> its last store), 8 / 9 the clock
  long long const t_task0 = gtime();
  constexpr int MOE_TRACE_ITEMS = 32;
  unsigned long long *const tr = trace + ((size_t)L.sm_id * 4 + (STATIC_TIDX >> 6)) * MOE_TRACE_ITEMS * 2;
  int n_tr = 0;
  auto trace_item = [&](long long t0, long long t1, int kind, int slot) {
    if ((STATIC_TIDX & 63) == 0 && n_tr < MOE_TRACE_ITEMS) {
      tr[2 * n_tr] = (unsigned long long)t0;
      tr[2 * n_tr + 1] = (unsigned long long)(t1 - t0) | ((unsigned long long)kind << 32) | ((unsigned long long)slot << 40);
    }
    n_tr++;
  };
#else
  (void)trace;
#endif
  int const own = xcc_id(), wave = STATIC_TIDX >> 6;
  unsigned dead = 0; // per set q, bit 8 q + x: queue x seen empty
  auto add = [&](int qx) {
    int v = 0;
    if ((STATIC_TIDX & 63) == 0) {
      v = (int)__hip_atomic_fetch_add(qs + 32 * qx, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    return __builtin_amdgcn_readfirstlane(v);
  };
  auto take_issue = [&](int q) -> int {
    if (dead & (1u << (8 * q + own))) {
      return -1;
    }
    int v = 0;
    if ((STATIC_TIDX & 63) == 0) {
      v = (int)__hip_atomic_fetch_add(qs + 32 * own, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    asm volatile("" ::: "memory");
    return v;
  };
  int w2_pre = -1;
  int n13q = 0, n2i = 0;
  auto take_finish = [&](int q, int n, int tk) -> int {
    int const n2 = n2i;
    auto u13 = [&](int x) { return n13q * (x + 1) / 8 - n13q * x / 8; };
    auto u2 = [&](int x) { return n2 * (x + 1) / 8 - n2 * x / 8; };
    if (tk != -1 || !(dead & (1u << (8 * q + own)))) {
      int const v = __builtin_amdgcn_readfirstlane(tk);
      int const U = u13(own);
      if (q == 0) {
        if (v < U) {
          return n13q * own / 8 + v;
        }
        dead |= 1u << own;
        if (v - U < u2(own)) {
          w2_pre = n2 * own / 8 + (v - U);
        } else {
          w2_pre = n2;
          dead |= 1u << (8 + own);
        }
        return n;
      }
      if (v >= U && v - U < u2(own)) {
        return n2 * own / 8 + (v - U);
      }
      dead |= 1u << (8 + own);
    }
    if (q == 0) {
      dead |= 1u << own;
      w2_pre = -1;
      return n;
    }
#if STATIC_CSR_W2_NOSTEAL
    // (STATIC_CSR_W2_NOSTEAL 1: a wave whose die's W2 items are all taken ends; no other die's counters read. MI355X dec456,
    // T 64: the last ~5 experts of each die's list ran 2-3x slower (20 vs 7 us per item) while every finished wave read the 8
    // counter lines to steal)
    return n;
#endif
    for (;;) {
      uint32_t c[8];
#pragma unroll
      for (int x = 0; x < 8; x++) {
        c[x] = (dead & (1u << (8 + x))) ? 0u : cnt_ld(qs + 32 * x);
      }
      int pick = -1;
      bool closed = false;
#pragma unroll
      for (int r = 1; r < 8; r++) {
        int const x = (own + r) & 7;
        if (dead & (1u << (8 + x))) {
          continue;
        }
        int const cx = (int)__builtin_amdgcn_readfirstlane(c[x]);
        if (cx < u13(x)) {
          closed = true;
          continue;
        }
        if (cx < u13(x) + u2(x)) {
          pick = x;
          break;
        }
        dead |= 1u << (8 + x);
      }
      if (pick < 0) {
        if (closed) {
          __builtin_amdgcn_s_sleep(32);
          continue;
        }
        return n;
      }
      int const v = add(pick);
      if (v >= u13(pick) && v - u13(pick) < u2(pick)) {
        return n2 * pick / 8 + (v - u13(pick));
      }
      dead |= 1u << (8 + pick);
    }
  };
  auto take_of = [&](int q, int n) -> int { return take_finish(q, n, take_issue(q)); };
  int const tk13 = wave == 0 ? take_issue(0) : -2; // (wave 0 takes the CU's W13 units)
  int const n_live = expert_table_csr<C>(pairs, sm);
#ifdef STATIC_NLIVE_STAMP
  if (L.sm_id == 0 && STATIC_TIDX == 0) { // (diagnosis: the experts used, CU 0's stamp slot STAMP_ISSUED, as moe_experts.cuh)
    g.stamps[STAMP_ISSUED] = n_live;
  }
#endif
  if (STATIC_TIDX == 0) {
    g.stamps[L.sm_id * NSTAMP + STAMP_TASK0] = gtime();
  }
#ifdef STATIC_TIMING_BUILD
  trace_item(t_task0, gtime(), 10, 0);
#endif
  int const *const eid = reinterpret_cast<int const *>(sm + Ls::EID), *const start = reinterpret_cast<int const *>(sm + Ls::START);
  unsigned char const *const ptok = reinterpret_cast<unsigned char const *>(sm + Ls::PTOK);
  int const n13 = n_live * C::G1, n_all = n13 + n_live * C::ITEMS2;
  int const nu = n13; // W13 units: (slot, 4 items), one CU's at a time
  n13q = nu;
  n2i = n_all - n13;
  // the CU's W13 units: wave 0 takes one (its next ticket issued once it is read), the CU stages the slot's z_q a K quarter
  // at a time, wave w computes item 4 g + w of the unit over all the slot's tokens, then their h_q
  int *const cuq = reinterpret_cast<int *>(sm + Ls::CUQ);
  int tkc = tk13;
#if STATIC_CSR_ZDMA
  bool zall = false; // (CU-uniform: every z_q block landed)
#endif
#pragma unroll 1
  for (;;) {
    __syncthreads(); // every wave read cuq and the staged z_q of the last unit
    if (wave == 0) {
      int const v = take_finish(0, nu, tkc);
      if ((STATIC_TIDX & 63) == 0) {
        *cuq = v;
      }
    }
    __syncthreads();
    int const jc = *cuq;
    if (jc >= nu) {
      break;
    }
    if (wave == 0) {
      tkc = take_issue(0);
    }
    int const slot = jc / C::G1, it = 4 * (jc % C::G1) + wave, e = eid[slot], s0 = start[slot], n = start[slot + 1] - s0;
#if STATIC_CSR_HQ_LOCAL
    // (STATIC_CSR_HQ_LOCAL 1, with NOSTEAL: a slot whose W13 units and W2 items all lie in this die's shares has every reader of its h_q
    // on this XCD - its h_q bytes by write-back stores (the XCD's L2; sc1 loads on the same XCD see them, amd/bench/xcd_poll.hip: 0.28
    // vs 0.40 us), the scales as before (12 B per pair: a line may span two slots))
    static_assert(STATIC_CSR_W2_NOSTEAL, "STATIC_CSR_HQ_LOCAL needs STATIC_CSR_W2_NOSTEAL");
    bool const hql = slot * C::G1 >= n13q * own / 8 && (slot + 1) * C::G1 <= n13q * (own + 1) / 8 &&
                     slot * C::ITEMS2 >= n2i * own / 8 && (slot + 1) * C::ITEMS2 <= n2i * (own + 1) / 8;
#else
    bool const hql = false;
#endif
#ifdef STATIC_TIMING_BUILD
    long long const t13 = gtime();
    long long t_st = 0;
#endif
    if (n <= 16) { // (CU-uniform)
      v4f acc1[4];
      v4i pf[W13_PF];
      expert_w13q_issue<C>(maps, sm, e, it, 0, pf); // steps 0 .. KQ - 1, the scales (in flight while z_q is staged)
#ifdef STATIC_TIMING_BUILD
      long long const ts0 = gtime();
#endif
      expert_stage_z16<C>(g, sm, ptok + s0, n);
      __syncthreads();
#ifdef STATIC_TIMING_BUILD
      t_st = gtime() - ts0;
#endif
      expert_w13s_run<C>(maps, sm, e, it, n, pf, acc1);
      expert_w13c_finish<C>(s0, n, it, acc1, acc1 + 2, hq, hsc, hql);
#ifdef STATIC_TIMING_BUILD
      trace_item(t13, gtime(), 1, slot);
      trace_item(t13, t13 + t_st, 14, slot);
#endif
      continue;
    }
    v4f acc[4][4];
#pragma unroll
    for (int c = 0; c < 4; c++)
#pragma unroll
      for (int t = 0; t < 4; t++) {
        acc[c][t] = v4f{0.f, 0.f, 0.f, 0.f};
      }
#pragma unroll 1
    for (int q = 0; q < 4; q++) {
      v4i pf[W13_PF];
      expert_w13q_issue<C>(maps, sm, e, it, q, pf); // (in flight while z_q is staged)
      if (q > 0) {
        __syncthreads(); // every wave's MFMAs of the last quarter read its staged z_q
      }
#ifdef STATIC_TIMING_BUILD
      long long const ts0 = gtime();
#endif
#if STATIC_CSR_ZDMA
      if (!zall) {
        expert_zq_all_wait<C>(g);
        zall = true;
      }
      expert_stage_zq_dma<C>(g, sm, q, ptok + s0, n);
#else
      expert_stage_zq<C>(g, sm, q, ptok + s0, n);
#endif
      __syncthreads();
#ifdef STATIC_TIMING_BUILD
      t_st += gtime() - ts0;
#endif
      expert_w13q_run<C>(sm, q, n, pf, acc);
    }
#pragma unroll
    for (int c = 0; c < 4; c++) {
      if (16 * c < n) {
        expert_w13c_finish<C>(s0 + 16 * c, n - 16 * c < 16 ? n - 16 * c : 16, it, acc[c], acc[c] + 2, hq, hsc, hql);
      }
    }
#ifdef STATIC_TIMING_BUILD
    trace_item(t13, gtime(), 1, slot);
    trace_item(t13, t13 + t_st, 14, slot);
#endif
  }
#ifdef STATIC_TIMING_BUILD
  long long const clk_t0 = gtime();
  long long const clk_c0 = (long long)__builtin_amdgcn_s_memtime();
#endif
  // W2 items: two in flight, A and B alternating (as moe_experts_task)
  int const n2 = n_all - n13, nu2 = n2;
  int ia = (w2_pre >= 0 && w2_pre < nu2) ? w2_pre : take_of(1, nu2);
  uint32_t *const rblk = qs + MOE_MISC_RBLK / 4;
  if (STATIC_TIDX == 0 && L.sm_id == 0) {
    __hip_atomic_store(rblk, (uint32_t)n_live, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  int pend_t0 = -1, pend_nt = 0;
  auto w2_rflush = [&]() {
    if (pend_t0 >= 0) {
      wait_mem();
      if ((STATIC_TIDX & 63) == 0) {
        for (int b = pend_t0 / 8; b <= (pend_t0 + pend_nt - 1) / 8; b++) {
          int const lo = pend_t0 > 8 * b ? pend_t0 : 8 * b, hi = pend_t0 + pend_nt < 8 * b + 8 ? pend_t0 + pend_nt : 8 * b + 8;
          __hip_atomic_fetch_add(rblk + RBLK_STRIDE * (1 + b), (uint32_t)(hi - lo), __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT);
        }
      }
      pend_t0 = -1;
    }
  };
  int pold_t0 = -1, pold_nt = 0;
  auto w2_lpend = [&](int t0_, int nt_) {
    if (pold_t0 >= 0) {
      int const st0 = pend_t0, snt = pend_nt;
      pend_t0 = pold_t0;
      pend_nt = pold_nt;
      w2_rflush();
      pend_t0 = st0;
      pend_nt = snt;
    }
    pold_t0 = pend_t0;
    pold_nt = pend_nt;
    pend_t0 = t0_;
    pend_nt = nt_;
  };
  auto w2_lflush_old = [&]() {
    if (pold_t0 >= 0 && (STATIC_TIDX & 63) == 0) {
      for (int b = pold_t0 / 8; b <= (pold_t0 + pold_nt - 1) / 8; b++) {
        int const lo = pold_t0 > 8 * b ? pold_t0 : 8 * b, hi = pold_t0 + pold_nt < 8 * b + 8 ? pold_t0 + pold_nt : 8 * b + 8;
        __hip_atomic_fetch_add(rblk + RBLK_STRIDE * (1 + b), (uint32_t)(hi - lo), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
    pold_t0 = -1;
  };
  auto ntok = [&](int u) { int const s = u / C::ITEMS2; return start[s + 1] - start[s]; };
  auto w2_load = [&](W2ItemC<C> &F, int u) {
    int const s = u / C::ITEMS2, n = ntok(u);
#ifdef STATIC_TIMING_BUILD
    F.t_load = gtime();
#endif
    expert_w2c_load<C>(F, maps, eid[s], (u % C::ITEMS2) * C::W2_TILES, start[s], n < 16 ? n : 16, hq, hsc);
  };
  auto w2_ready = [&](W2ItemC<C> &F, int u) {
    int const n = ntok(u);
    expert_w2c_ready<C>(F, start[u / C::ITEMS2], n < 16 ? n : 16, hq, hsc);
  };
  auto w2_compute = [&](W2ItemC<C> &F, int u, auto &&hook) {
    int const s = u / C::ITEMS2, t0 = (u % C::ITEMS2) * C::W2_TILES, s0 = start[s], n = ntok(u);
    expert_w2c_compute<C>(F, sm, s0, n < 16 ? n : 16, t0, out, hook);
#pragma unroll 1
    for (int c0 = 16; c0 < n; c0 += 16) { // (a slot of more than 16 tokens: the same weights, the next 16)
      int const nc = n - c0 < 16 ? n - c0 : 16;
      expert_w2c_hq<C>(F, s0 + c0, nc, hq, hsc);
      expert_w2c_ready<C>(F, s0 + c0, nc, hq, hsc);
      expert_w2c_compute<C>(F, sm, s0 + c0, nc, t0, out);
    }
    w2_lpend(t0, C::W2_TILES);
#ifdef STATIC_TIMING_BUILD
    trace_item(F.t_load, gtime(), 2, s);
#endif
  };
  if (ia < nu2) {
    W2ItemC<C> A, B;
    int tk = take_issue(1);
    w2_load(A, ia);
    w2_ready(A, ia);
#pragma unroll 1
    for (;;) {
      int const ib = take_finish(1, nu2, tk);
      if (ib >= nu2) {
        w2_compute(A, ia, W2NoHook{});
        break;
      }
      tk = take_issue(1);
      w2_load(B, ib);
      w2_compute(A, ia, W2NoHook{});
      w2_ready(B, ib);
      w2_lflush_old();
      ia = take_finish(1, nu2, tk);
      if (ia >= nu2) {
        w2_compute(B, ib, W2NoHook{});
        break;
      }
      tk = take_issue(1);
      w2_load(A, ia);
      w2_compute(B, ib, W2NoHook{});
      w2_ready(A, ia);
      w2_lflush_old();
    }
  }
  wait_mem();
  if (pold_t0 >= 0) {
    w2_lflush_old();
  }
  w2_rflush();
  if (STATIC_TIDX == 0) {
    g.stamps[L.sm_id * NSTAMP + STAMP_TASK1] = gtime();
  }
#ifdef STATIC_TIMING_BUILD
  {
    long long const clk_c1 = (long long)__builtin_amdgcn_s_memtime();
    trace_item(clk_t0, gtime(), 8, 0);
    trace_item(clk_t0, clk_t0 + ((clk_c1 - clk_c0) & 0xFFFFFFFFll), 9, 0);
  }
#endif
  __syncthreads();
}

} // namespace static_mk
