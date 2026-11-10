// tasks/glm_moe_experts.cuh (HIP, gfx950) -- GLM-5.3-Flash's MoE experts on this GPU's TP shard (IR = the expert
// intermediate / GPUs features of every expert), the shared expert in the same queue. One task per CU:
//   0. every CU builds the same routing table from glm_topk_route's pairs ([T][KP]: expert << 32 | weight bits,
//      0xFF-prefilled, polled): the experts used ("slots", in ascending id), each slot's pairs
//   1. a queue (one counter) the waves take items from, each wave its own: first the W13 units (slot, 16 features: the 16
//      gate rows, then the 16 up rows), then the W2 units (slot, 32 output rows); a W2 unit waits for its slot's IR / 16
//      W13 units (a counter per slot). Every W13 unit is handed out before any W2 unit, and a W2 unit waits only for W13
//      units already taken: no deadlock.
//   W13 unit: x (bf16 [T][H], the tokens of the slot, 16 per MFMA column tile) x the unit's 32 weight rows (e4m3, the
//      checkpoint's bytes): each e4m3 value turned into bf16 exactly in registers, bf16 MFMAs 16x16x32, one sum per 128-K
//      block multiplied by the block's fp32 scale (the checkpoint's weight_scale_inv) and added: the weights exact, no
//      activation quantized. h = SiLU(min(gate, limit)) *
//      clamp(up, -limit, limit) (SGLang's swiglu_clamped; no limit: SiLU(gate) up) -> bf16 h [pair][IR] (scratch)
//   W2 unit: h (the slot's pairs) x the 128 rows (e4m3, K = IR) the same way, x the pair's routing weight -> R [pair][H]
//      fp32 (an allreduce_send node adds a token's KP rows and sends the bf16 sum to every GPU)
// Weights: W13 [NB][IR / 16][32 rows][H] e4m3 (unit u: gate rows of features 16 u .. 16 u + 15, then their up rows; a
//          unit's bytes as [H / 128][2 (gate, up)][2 halves][64 lanes][16 B]: lane 16 grp + r16 half h holds row r16, K
//          128 kb + 32 grp + 16 h ..: each wave load 1 KB contiguous; W2's 16-row tiles likewise [IR / 128][2][64][16 B]),
//          W13 scales [NB][IR / 128][H / 128][2] fp32 (gate, up); W2 [NB][H / 128][128 rows][IR] e4m3, W2 scales
//          [NB][H / 128][IR / 128] fp32 (amd/tests/test_glm_moe_layer.py pack_experts)
#pragma once
#include "../core.cuh"
#include <type_traits>

// STATIC_GME_DBG (debug): CU b writes its phase into dbg[b] (buffer slot 6: a pinned host tensor the harness reads during a
// hang): 1 the table cleared, 2 its pairs landed, 3 its table built, 4 + n: n queue items taken, 1000 done
#ifndef STATIC_GME_DBG
#define STATIC_GME_DBG 0
#endif
// STATIC_GME_WBL2 (0 default; amd/serving/glm/build_glm_libs.sh sets 1 in gkdamoe T >= 32, gkdamoe3 T >= 16): at the experts task's start (while the table waits for the routing), CUs 0 .. 7 write back
// their XCD's L2 (buffer_wbl2): in the one-kernel layer the KDA part's state stores leave dirty lines the experts' weight
// stream would otherwise evict one by one (MI355X 11-08 glmpair7: experts +14 us after the KDA part, +3 without state stores)
#ifndef STATIC_GME_WBL2
#define STATIC_GME_WBL2 0
#endif
namespace static_mk {
namespace gme {
constexpr int GME_SS = 4;    // counters between two slots' W13 counts (16 B)
// output rows of a wave's W2 unit (whole MFMA tile pairs): small at small T, so every wave has one
constexpr int W2_ROWS = T <= 4 ? 32 : T <= 16 ? 64 : 128;
// STATIC_GME_TIMES (diagnosis): per wave gw (= wave NCU + cu) 4 times (gtime ns) into dbg as long long [4 NCU][4]: the
// table built, its first W13 item's loads issued, that item's sums done, its W13 items' end counted (dbg: a device tensor)
#ifndef STATIC_GME_TIMES
#define STATIC_GME_TIMES 0
#endif
// Slots: an expert with c tokens takes ceil(c / 16) slots of at most 16 tokens (slot j: its pairs
// 16 j ..): every item one token tile (NT 1, x of the whole chunk with its weights), the shared expert's units spread over
// 4 slots at T 64 (its weights read once per slot). MI355X 11-07 glmmoe47 / 49: the shared expert's 4-tile items took
// 58-85 us vs ~20 us for 1 tile
template <int NB_, int KP_, int H_, int IR_, int KQ_>
struct Cfg {
  static constexpr int NB = NB_, KP = KP_, H = H_, IR = IR_, KQ = KQ_; // KQ: K pieces of a W13 unit (waves)
  static constexpr int NPAIR = T * KP, NSLOT = (NB < NPAIR ? NB : NPAIR) + NPAIR / 16;
  static constexpr int U13 = IR / 16, U2 = H / 128, KB1 = H / 128, KB2 = IR / 128, IRB = IR / 128;
  static constexpr int NT = 1; // token tiles of a slot (at most 16 tokens)
  // ucnt: KQ > 1 the units' piece counts [NSLOT][U13] first; then per slot its units done (a 128-B line each) at SC0 + 32 s
  static constexpr int SC0 = KQ > 1 ? (NSLOT * U13 + 31) / 32 * 32 : 0;
  static_assert(H % 512 == 0 && IR % 128 == 0 && KB1 % 4 == 0 && KB2 <= 4 && T <= 64 && KB1 % (KQ * 2) == 0,
                "glm_moe_experts: the shapes");
};

// the pipelined W13 phase (T <= 8: x in LDS, the next chunk's weights loaded while this one's MFMAs run)
// (T 16 would need x 131 KB of LDS, KQ 1)
template <class C>
constexpr bool PIPE = T <= 8;
// CUW (T <= 8): a W13 unit is a CU's - its 4 waves a quarter of K each, their sums added in LDS
// (wave 0 writes h); the CU's units cu, cu + NCU, .. with the next unit's weights loaded during this one's MFMAs.
// MI355X 11-07 glmmoe28 (T 1, K pieces over the GPU): the pieces' hand-off through memory (partial stores, a counter
// atomic, partial loads, the h store: 4 round trips in a row) ended W13 ~10-14 us after the sums on the late waves
// CUW: the shared expert (every token, slot 0) needs no routing - CUs
// 0 .. U13 - 1 compute its W13 units (unit = cu) before the table, while the router / topk run; the routed units then start at
// CU 0 (unit U13 + k on CU k % NCU). MI355X 11-08 glmmoe123 (T 1, 288 units on 256 CUs): W13 ended at 20.5 us (median) but
// 28.5 on CUs 0 .. 31 (a second unit), every W2 item of that slot after them; experts ended 33.4. glmmoe124 (A B A B, checks pass):
// MoE T 1 49.8 / 49.4 vs 54.8 / 54.0 us, T 2 / 4 level, T 8 105.1 / 106.4 vs 109.3 / 111.2; pair T 1 86.6 / 86.9 vs 89.6 / 87.6,
// T 2 98.9 / 100.8 vs 102.1 / 100.0, T 8 168.0 / 168.1 vs 168.8 / 170.6
template <class C>
constexpr bool CUW = PIPE<C>;
// SLOTW (the generic path, T > 8): a W2 item waits for its slot's W13 units only (per-slot counts), not for
// every W13 unit. A wave posts a finished unit's count at its next item's end (that item's loads were consumed, and vector
// memory operations complete in issue order, so the unit's h stores are done) or when its W13 items end. MI355X 11-07
// glmmoe41 (random x, T 64): W13 ends per wave 172 (5 %) .. 225 us, every W2 item waited for 227
// CUW: x [T][H] into the LDS before the routing table (x does not depend on the
// routing; its loads overlap the router / topk), instead of after it. MI355X 11-08 glmmoe90 (random x, A B A B): T 8 115.9 /
// 118.7 vs 120.3 / 122.3 us, T 4 87.4 / 88.6 vs 90.4 / 91.2, T 1 56.7 / 60.0 vs 59.0 / 57.6
// DYP (T >= 32: 2, else 0; the last DYP x 4 NCU W2 items): the W2 items are taken round robin up to the last DYP rounds' worth; those are handed out one at a time by a counter (the node's counter line 32, 0 each launch),
// so the waves that finish early take more (with it the W2 weighting, W2BAL, is off). MI355X 11-08 glmmoe100 (random x): T 64
// W2 ended at 186 / 201 / 225 us (5 % / median / max), T 32 130 / 140 / 162 - the sends wait for the last
// (T 16 was 1 until glmmoe122: 0 -4..-8 us with RBLK + hand-off. MI355X 11-08 glmmoe102, random x, A B C D x 2: T 64 N 0 / 1 / 2 / 4
// 276.8 / 269.0 / 263.4 / 268.6 and 274.2 / 270.3 / 266.4 / 270.4 us; T 32 202.3 / 197.5 / 196.6 / 201.7, 200.8 / 198.9 / 199.4 /
// 200.9; T 16 168.7 / 165.4 / 170.1 / 181.0, 166.9 / 166.1 / 169.1 / 183.3)
// R in blocks (the host's MPK_GME_RBLK, default 1): K3's R-in-blocks protocol for the sends - CU 0 stores the slot count
// in the header (word 0), the W2 items run by column chunk (u-major: item j = slot j mod ns, chunk j / ns) and each adds
// 8 W2_ROWS / 128 to its 128-column block's counter (word RBLK_STRIDE (1 + b)) once its R stores are done; allreduce_send
// (params[2] 2) waits only for the blocks its slice reads, so the first chunks' sums go out while W2 runs. MI355X 11-08
// glmmoe99 / 100: T 64 the sends waited for the last W2 item (~217 us), the tail to hc_pre ~37 us
// (MI355X 11-08 glmmoe105, random x, A B A B: T 64 266.4 / 268.6 vs 264.7 / 265.2 us, T 32 198.9 / 201.4 vs 193.3 /
// 193.5, T 16 168.2 / 164.0 vs 166.9 / 160.2, T 8 119.0 / 120.2 vs 107.7 / 111.8)
template <class C>
constexpr bool SLOTW = !PIPE<C>; // (PIPE: the next item's weights are in flight at an item's end)

// LDS (bytes)
template <class C>
struct Lds {
  static constexpr int PE = 0;                              // [NPAIR] int: pair p's expert
  static constexpr int PW = PE + C::NPAIR * 4;              // [NPAIR] float: its weight
  static constexpr int CNT = PW + C::NPAIR * 4;             // [NB] int: pairs per expert, then the fill
  static constexpr int SOF = CNT + C::NB * 4;               // [NB] int: expert e's slot (-1: unused)
  static constexpr int SE = SOF + C::NB * 4;                // [NSLOT] int: slot s's expert
  static constexpr int START = SE + C::NSLOT * 4;           // [NSLOT + 1] int
  static constexpr int PL = START + (C::NSLOT + 1) * 4;     // [NPAIR] int: the pairs, grouped by slot
  static constexpr int SLL = PL + C::NPAIR * 4;             // [NB] int: expert e's slot within its group of 64
  static constexpr int STL = SLL + C::NB * 4;               // [NB] int: its pairs' start within the group
  static constexpr int GT = STL + C::NB * 4;                // [16] int: per group of 64 experts its slots, its pairs
  static constexpr int MISC = (GT + 64 + 15) / 16 * 16;     // [4] int: the slot count, the W13-done flag, the poller
  static constexpr int XS = MISC + 16;                      // PIPE: x [T][H + 8] bf16 (rows 16 B apart in the banks)
  static constexpr int PR = XS + (PIPE<C> ? T * (C::H + 8) * 2 : 0); // CUW: [2][4 waves][2][64 lanes][4] fp32
  static constexpr int BYTES = PR + (CUW<C> ? 2 * 4 * 2 * 64 * 4 * 4 : 0);
};

// 8 e4m3 bytes (OCP, gfx950's v_cvt_pk_f32_fp8) -> 8 bf16 (exact)
__device__ __forceinline__ v8bf e4m3x8_bf16(uint32_t lo, uint32_t hi) {
  v8bf r;
  uint32_t const w[2] = {lo, hi};
#pragma unroll
  for (int i = 0; i < 2; i++) {
    auto const a = __builtin_amdgcn_cvt_pk_f32_fp8((int)w[i], false); // bytes 0, 1
    auto const b = __builtin_amdgcn_cvt_pk_f32_fp8((int)w[i], true);  // bytes 2, 3
    r[4 * i + 0] = (__bf16)a[0];
    r[4 * i + 1] = (__bf16)a[1];
    r[4 * i + 2] = (__bf16)b[0];
    r[4 * i + 3] = (__bf16)b[1];
  }
  return r;
}
__device__ __forceinline__ v8bf u4_bf16(uint4 const &u) {
  return __builtin_bit_cast(v8bf, u);
}

} // namespace gme

template <class C>
__device__ __forceinline__ void glm_moe_experts_task(G const &g,
                                                     __hip_bfloat16 const *x,         // [T][H]
                                                     uint32_t const *x_cnt,           // x's producer's counter (or null)
                                                     uint32_t x_need,
                                                     unsigned long long const *pairs, // [T][KP]
                                                     uint8_t const *w13,
                                                     float const *w13sf,
                                                     uint8_t const *w2,
                                                     float const *w2sf,
                                                     __hip_bfloat16 *hbuf,            // [NPAIR][IR] scratch
                                                     float *part,                     // [KQ][NPAIR][2][IR] (KQ > 1)
                                                     uint32_t *ucnt,                  // 0 each launch (Cfg::SC0)
                                                     float *R,                        // [NPAIR][H]
                                                     uint32_t *cnt,                   // the node's counters
                                                     uint32_t *rblk,                  // (R in blocks: the blocks' counters, 0 each launch)
                                                     float limit,
                                                     int *dbg,
                                                     int cu,  // this task (one per CU) of ncu
                                                     int ncu,
                                                     char *sm STAGE_STAMP_PARAM) {
  using namespace gme;
  using L = Lds<C>;
  constexpr int NPAIR = C::NPAIR, NB = C::NB, KP = C::KP, H = C::H, IR = C::IR, NT = C::NT;
  int const tid = STATIC_TIDX, lane = tid & 63, wave = __builtin_amdgcn_readfirstlane(tid >> 6), r16 = lane & 15,
            grp = lane >> 4;
  int *pe = reinterpret_cast<int *>(sm + L::PE);
  float *pw = reinterpret_cast<float *>(sm + L::PW);
  int *cntl = reinterpret_cast<int *>(sm + L::CNT);
  int *sof = reinterpret_cast<int *>(sm + L::SOF);
  int *se = reinterpret_cast<int *>(sm + L::SE);
  int *start = reinterpret_cast<int *>(sm + L::START);
  int *pl = reinterpret_cast<int *>(sm + L::PL);
  int *misc = reinterpret_cast<int *>(sm + L::MISC);
  // (MI355X 11-07: W2 units waiting on per-slot counters, ~2800 waves polling one line, starved the W13 units' loads: a
  // hang. Now a W2 unit waits for every W13 unit, one wave per CU polls the count, the others an LDS flag)
  uint32_t *w13done = cnt + 64; // the CUs whose waves finished all their W13 items (its own line)
  __syncthreads(); // the LDS may still be read by the previous task
  if constexpr (STATIC_GME_WBL2) { // (CUs 0 .. 7, one per XCD when WGs go round robin: the XCD's dirty L2 lines written back now)
    if (cu < 8 && tid == 0) {
      asm volatile("buffer_wbl2 sc1\n s_waitcnt vmcnt(0)" ::: "memory");
    }
  }
  auto x_to_lds = [&]() { // (CU-unit path: x [T][H] -> LDS rows of H + 8)
    if constexpr (CUW<C>) {
      __hip_bfloat16 *const xs = reinterpret_cast<__hip_bfloat16 *>(sm + L::XS);
      for (int v = tid; v < T * H / 8; v += 256) {
        int const t = v / (H / 8), c8 = 8 * (v % (H / 8));
        *reinterpret_cast<uint4 *>(xs + t * (H + 8) + c8) = ld16_in_at(x, (uint32_t)(((size_t)t * H + c8) * 2));
      }
    }
  };
  if constexpr (CUW<C>) { // (x first: its producer's count, then the loads)
    if (x_cnt != nullptr && tid == 0) {
      cnt_wait_in(x_cnt, x_need);
    }
    __syncthreads();
    x_to_lds();
  }
  if constexpr (CUW<C>) { // (the shared expert's unit cu, tokens 0 .. T - 1, pairs t KP + KP - 1)
    static_assert(C::KQ == 1 && T <= 16, "glm_moe_experts CUW: x in LDS first, the shared expert in slot 0 (one slot), whole units");
    __syncthreads(); // (x in LDS)
    if (cu < C::U13) {
      __hip_bfloat16 const *const xs = reinterpret_cast<__hip_bfloat16 const *>(sm + L::XS);
      constexpr int KBW = C::KB1 / 4, XR = H + 8;
      int const e = NB - 1, u = cu, c0 = wave * KBW;
      uint8_t const *wbp = w13 + ((size_t)e * C::U13 + u) * 32 * H;
      uint4 w[2][KBW][2];
      float sg[KBW], su[KBW];
#pragma unroll
      for (int i = 0; i < 2; i++)
#pragma unroll
        for (int kb = 0; kb < KBW; kb++) {
          uint8_t const *rp = wbp + ((size_t)((c0 + kb) * 2 + i) * 2) * 1024 + 16 * lane;
          w[i][kb][0] = *reinterpret_cast<uint4 const *>(rp);
          w[i][kb][1] = *reinterpret_cast<uint4 const *>(rp + 1024);
        }
#pragma unroll
      for (int kb = 0; kb < KBW; kb++) {
        size_t const so = (((size_t)e * C::IRB + u / 8) * C::KB1 + c0 + kb) * 2;
        sg[kb] = w13sf[so];
        su[kb] = w13sf[so + 1];
      }
      int const tok = r16 < T ? r16 : T - 1;
      v4f tt2[2] = {v4f{0.f, 0.f, 0.f, 0.f}, v4f{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
      for (int kb = 0; kb < KBW; kb++) {
        v4f acc[2] = {v4f{0.f, 0.f, 0.f, 0.f}, v4f{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
        for (int st = 0; st < 4; st++) {
          v8bf const b = u4_bf16(*reinterpret_cast<uint4 const *>(xs + tok * XR + (c0 + kb) * 128 + 32 * grp + 8 * st));
#pragma unroll
          for (int i = 0; i < 2; i++) {
            uint4 const &wv = w[i][kb][st >> 1];
            v8bf const a = (st & 1) ? e4m3x8_bf16(wv.z, wv.w) : e4m3x8_bf16(wv.x, wv.y);
            acc[i] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, acc[i], 0, 0, 0);
          }
        }
        tt2[0] += acc[0] * sg[kb];
        tt2[1] += acc[1] * su[kb];
      }
      float *const pb = reinterpret_cast<float *>(sm + L::PR) + (size_t)1 * (4 * 2 * 64 * 4); // (buffer 1: the main rounds start on 0)
#pragma unroll
      for (int i = 0; i < 2; i++) {
        *reinterpret_cast<v4f *>(pb + ((wave * 2 + i) * 64 + lane) * 4) = tt2[i];
      }
      __syncthreads();
      if (wave == 0) {
        v4f tot2[2];
#pragma unroll
        for (int i = 0; i < 2; i++) {
          tot2[i] = *reinterpret_cast<v4f const *>(pb + ((0 * 2 + i) * 64 + lane) * 4);
#pragma unroll
          for (int ww = 1; ww < 4; ww++) {
            tot2[i] += *reinterpret_cast<v4f const *>(pb + ((ww * 2 + i) * 64 + lane) * 4);
          }
        }
        if (r16 < T) {
          float hv[4];
#pragma unroll
          for (int q = 0; q < 4; q++) {
            float gv = tot2[0][q], uv = tot2[1][q];
            if (limit > 0.f) {
              gv = fminf(gv, limit);
              uv = fminf(fmaxf(uv, -limit), limit);
            }
            hv[q] = gv / (1.0f + __expf(-gv)) * uv;
          }
          int const p = r16 * KP + KP - 1;
          uint2 const hw = make_uint2(pack_bf16x2(hv[0], hv[1]), pack_bf16x2(hv[2], hv[3]));
          __hip_atomic_store(reinterpret_cast<unsigned long long *>(hbuf + (size_t)p * IR + 16 * u + 4 * grp),
                             ((unsigned long long)hw.y << 32) | hw.x, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        wait_mem();
        if (lane == 0) { // (slot 0's count: the shared expert's slot with LASTFIRST)
          __hip_atomic_fetch_add(ucnt + C::SC0, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
      }
    }
  }
  // 0. the routing table
  for (int e = tid; e < NB; e += 256) {
    cntl[e] = 0;
  }
  __syncthreads();
  if (STATIC_GME_DBG && tid == 0) {
    __hip_atomic_store(dbg + blockIdx.x, 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  for (int p = tid; p < NPAIR; p += 256) {
    unsigned long long v;
    for (;;) {
      v = __hip_atomic_load(pairs + p, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      if (v != ~0ull) {
        break;
      }
      poll_pause();
    }
    int const e = min(max((int)(v >> 32), 0), NB - 1); // (in the bank whatever the routing gave)
    pe[p] = e;
    pw[p] = __uint_as_float((uint32_t)v);
    atomicAdd(&cntl[e], 1);
  }
  __syncthreads();
  if (STATIC_GME_DBG && tid == 0) {
    __hip_atomic_store(dbg + blockIdx.x, 2, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  // slots in ascending expert id: per group of 64 experts (a wave) a ballot (slots) and a scan (pairs), then the groups
  // before (MI355X 11-07: the serial loop of thread 0 over the 289 experts took ~17 us)
  int *sll = reinterpret_cast<int *>(sm + L::SLL);
  int *stl = reinterpret_cast<int *>(sm + L::STL);
  int *gt = reinterpret_cast<int *>(sm + L::GT);
  constexpr int NG = (NB + 63) / 64;
  static_assert(NG <= 8, "glm_moe_experts: at most 512 experts");
  for (int gi = wave; gi < NG; gi += 4) {
    int const e = 64 * gi + lane, c = e < NB ? cntl[e] : 0;
    int const ks = (c + 15) >> 4; // (its slots)
    unsigned long long const m = __ballot(c > 0);
    int inc = c, ink = ks;
#pragma unroll
    for (int d = 1; d < 64; d *= 2) {
      int const v = __shfl_up(inc, d), vk = __shfl_up(ink, d);
      if (lane >= d) {
        inc += v;
        ink += vk;
      }
    }
    if (e < NB) {
      sll[e] = ink - ks; // (its first slot within the group)
      stl[e] = inc - c;
    }
    if (lane == 63) {
      gt[gi] = ink;
      gt[8 + gi] = inc;
    }
    (void)m;
  }
  __syncthreads();
  // Expert NB - 1 (the shared expert: every token) takes slot 0, the others move up one slot and their
  // pairs by its count (its items, the slot with the most tokens, run in the first round, not the last). MI355X 11-07
  // glmmoe46 (random x, T 64): its 32 W13 units were items 5184.. (waves 64..95, 6th round), 4 token tiles each: those
  // waves ended W13 at 210-218 us, every other wave at 138-149
  int const csh = cntl[NB - 1], ksh = (csh + 15) >> 4; // (its slots)
  for (int e = tid; e < NB; e += 256) {
    int const gi = e >> 6;
    int os = 0, oc = 0;
    for (int j = 0; j < gi; j++) {
      os += gt[j];
      oc += gt[8 + j];
    }
    int sl = os + sll[e], st0 = oc + stl[e];
    if (csh > 0) {
      sl = e == NB - 1 ? 0 : sl + ksh;
      st0 = e == NB - 1 ? 0 : st0 + csh;
    }
    int const c = cntl[e], ks = (c + 15) >> 4;
    sof[e] = c > 0 ? sl : -1;
    for (int j = 0; j < ks; j++) { // (slot j of the expert: its pairs 16 j ..)
      se[sl + j] = e;
      start[sl + j] = st0 + 16 * j;
    }
  }
  if (tid == 0) {
    int ns = 0, at = 0;
    for (int j = 0; j < NG; j++) {
      ns += gt[j];
      at += gt[8 + j];
    }
    start[ns] = at;
    misc[0] = ns;
    misc[1] = 0; // this CU's waves past their W13 items
    misc[2] = 0; // every W13 unit done (seen by this CU's poller)
    misc[3] = 0; // this CU's poller taken
  }
  __syncthreads();
  for (int e = tid; e < NB; e += 256) {
    cntl[e] = 0; // the fill below
  }
  __syncthreads();
  for (int p = tid; p < NPAIR; p += 256) {
    int const e = pe[p], s = sof[e];
    pl[start[s] + atomicAdd(&cntl[e], 1)] = p;
  }
  __syncthreads();
  // A split expert's pairs (more than 16) in pair order, the same on every CU (the fill above orders them by its
  // atomics, which differ per CU: CU A's slot 0 and CU B's slot 0 held different tokens, some pairs' h never computed -
  // MI355X 11-07 glmmoe53 / 54: 28 of the shared expert's 32 pairs wrong at T 32, CU 0's table itself right)
  // (wave w: experts 64 g + lane of groups g = w, w + 4, ..; the split ones in turn. MI355X 11-07 glmmoe55: a loop over
  // every expert per wave added ~4 us to the table)
  for (int gi = wave; gi < (NB + 63) / 64; gi += 4) {
   int const el = 64 * gi + lane;
   unsigned long long sm_ = __ballot(el < NB && cntl[el] > 16); // (after the fill: the counts again)
   while (sm_ != 0ull) {
    int const e = 64 * gi + __builtin_ctzll(sm_);
    sm_ &= sm_ - 1ull;
    int const b = start[sof[e]];
    int base = 0;
    for (int q0 = 0; q0 < NPAIR; q0 += 64) {
      int const q = q0 + lane;
      bool const m = q < NPAIR && pe[q] == e;
      unsigned long long const bm = __ballot(m);
      if (m) {
        pl[b + base + __popcll(bm & ((1ull << lane) - 1ull))] = q;
      }
      base += __popcll(bm);
    }
   }
  }
  __syncthreads();
  // (wave-uniform values read from LDS: readfirstlane, so the branches below with their barriers are uniform control flow -
  // read as divergent, the compiler made the waves of a CU take different counts of barriers: a hang)
  constexpr int KQ = C::KQ;
  int const ns = __builtin_amdgcn_readfirstlane(misc[0]), n13u = ns * C::U13, n13 = n13u * KQ; // units, pieces
  bool w13_seen = false; // this wave knows every W13 unit is done
  if (STATIC_GME_DBG && tid == 0) {
    __hip_atomic_store(dbg + blockIdx.x, 3, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
  if (tid == 0) { // (timing build: stage stamp 0 the table built, 1 this CU's poller saw every W13 unit done)
    STAGE_STAMP_AT(0)
  }
  if (cu == 0 && tid == 0) {
    __hip_atomic_store(rblk, (uint32_t)ns, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  // 1. the queue, per wave (no block barriers below: each wave takes its own items; MI355X 11-07: with the queue per CU,
  // its barriers inside exec-mask-structurized branches, the waves of a CU took different counts of barriers and hung)
  if (x_cnt != nullptr) {
    if (lane == 0) {
      cnt_wait_in(x_cnt, x_need);
    }
  }
  constexpr int U2W = H / W2_ROWS; // W2 units (one wave's) per expert
  // static items: wave gw = wave NCU + cu takes items gw, gw + 4 NCU, .. (every CU has the same table; the items of a phase
  // are the same size). MI355X 11-07: a queue (one fetch_add per item on one counter) serialized ~13 ns per item: W13 ~31
  // us fixed at T 1 (~3300 items + ends), ~200 us at T 64. The W13 phase's end: a CU's waves count in LDS, its last adds 1
  // (NCU adds, not one per unit)
  int const nw = 4 * ncu, gw = wave * ncu + cu, ntot = n13 + ns * U2W;
  bool w13_posted = false;
  long long *const tms = reinterpret_cast<long long *>(dbg) + 8 * gw; // (STATIC_GME_TIMES: 8 per wave)
  int n2done = 0;
  if (STATIC_GME_TIMES && lane == 0) {
    tms[0] = gtime();
  }
  // a W13 item's end: KQ > 1: the piece's sums to part; the unit's last piece (its wave) adds the KQ pieces in order (the
  // same sums every launch) and goes on to h; h = swiglu -> hbuf
  int pend = -1; // (SLOTW: the slot of this wave's last finished unit, its count not posted yet)
  auto post_pend = [&]() {
    if constexpr (SLOTW<C>) {
      if (pend >= 0) {
        wait_mem(); // (nothing in flight here but the unit's h stores)
        if (lane == 0) {
          __hip_atomic_fetch_add(ucnt + C::SC0 + 32 * pend, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        pend = -1;
      }
    }
  };
  auto finish13 = [&](v4f (&tot)[2][NT], int s, int u, int kq, int p0, int n, int it) {
      if (STATIC_GME_TIMES && lane == 0 && it == gw) {
        tms[2] = gtime();
      }
      post_pend();
      bool last = true;
      if constexpr (KQ > 1) {
        int const un = s * C::U13 + u;
#pragma unroll
        for (int tt = 0; tt < NT; tt++) {
          int const j = 16 * tt + r16;
          if (j < n) {
            int const p = pl[p0 + j];
#pragma unroll
            for (int i = 0; i < 2; i++) {
              st16_out(part + (((size_t)kq * NPAIR + p) * 2 + i) * IR + 16 * u + 4 * grp,
                       make_uint4(__float_as_uint(tot[i][tt][0]), __float_as_uint(tot[i][tt][1]),
                                  __float_as_uint(tot[i][tt][2]), __float_as_uint(tot[i][tt][3])));
            }
          }
        }
        wait_mem();
        uint32_t old = 0;
        if (lane == 0) {
          old = __hip_atomic_fetch_add(ucnt + un, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        last = __builtin_amdgcn_readfirstlane((int)__shfl((int)old, 0)) == KQ - 1;
        asm volatile("" ::: "memory");
        if (last) {
#pragma unroll
          for (int tt = 0; tt < NT; tt++) {
            int const j = 16 * tt + r16;
            int const p = j < n ? pl[p0 + j] : 0;
#pragma unroll
            for (int i = 0; i < 2; i++) {
              v4f a = v4f{0.f, 0.f, 0.f, 0.f};
#pragma unroll
              for (int q = 0; q < KQ; q++) {
                uint4 const w = ld16_in_at(part, (uint32_t)(((((size_t)q * NPAIR + p) * 2 + i) * IR + 16 * u + 4 * grp) * 4));
                a += v4f{__uint_as_float(w.x), __uint_as_float(w.y), __uint_as_float(w.z), __uint_as_float(w.w)};
              }
              tot[i][tt] = a;
            }
          }
        }
      }
      // h of features 16 u + 4 grp .. + 3 (rows of the MFMA's D), token 16 tt + r16
#pragma unroll
      for (int tt = 0; tt < NT; tt++) {
        int const j = 16 * tt + r16;
        if (last && j < n) {
          float hv[4];
#pragma unroll
          for (int r = 0; r < 4; r++) {
            float gv = tot[0][tt][r], uv = tot[1][tt][r];
            if (limit > 0.f) {
              gv = fminf(gv, limit);
              uv = fminf(fmaxf(uv, -limit), limit);
            }
            hv[r] = gv / (1.0f + __expf(-gv)) * uv;
          }
          int const p = pl[p0 + j];
          uint2 const hw = make_uint2(pack_bf16x2(hv[0], hv[1]), pack_bf16x2(hv[2], hv[3]));
          __hip_atomic_store(reinterpret_cast<unsigned long long *>(hbuf + (size_t)p * IR + 16 * u + 4 * grp),
                             ((unsigned long long)hw.y << 32) | hw.x, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
      }
      if (last) {
        pend = s;
      }
  };
  int it0 = gw; // this wave's first item of the generic loop below
  if constexpr (CUW<C>) {
    static_assert(KQ == 1, "glm_moe_experts: CUW splits K over the CU's waves (KQ 1)");
    __hip_bfloat16 *const xs = reinterpret_cast<__hip_bfloat16 *>(sm + L::XS);
    constexpr int XR = H + 8;
    __syncthreads();
    constexpr int KBW = C::KB1 / 4; // K blocks of a wave (H 4096: 8, one 32-KB chunk)
    struct WSet {
      uint4 w[2][KBW][2];
      float sg[KBW], su[KBW];
    };
    float *const prs = reinterpret_cast<float *>(sm + L::PR);
    WSet wa, wb2;
    auto issue = [&](WSet &ws, int un) {
      int const s = un / C::U13, u = un % C::U13, e = __builtin_amdgcn_readfirstlane(se[s]), c0 = wave * KBW;
      uint8_t const *wbp = w13 + ((size_t)e * C::U13 + u) * 32 * H;
#pragma unroll
      for (int i = 0; i < 2; i++)
#pragma unroll
        for (int kb = 0; kb < KBW; kb++) {
          uint8_t const *rp = wbp + ((size_t)((c0 + kb) * 2 + i) * 2) * 1024 + 16 * lane;
          ws.w[i][kb][0] = *reinterpret_cast<uint4 const *>(rp);
          ws.w[i][kb][1] = *reinterpret_cast<uint4 const *>(rp + 1024);
        }
#pragma unroll
      for (int kb = 0; kb < KBW; kb++) {
        size_t const so = (((size_t)e * C::IRB + u / 8) * C::KB1 + c0 + kb) * 2;
        ws.sg[kb] = w13sf[so];
        ws.su[kb] = w13sf[so + 1];
      }
    };
    // one round: this wave's quarter of unit un, its sums to LDS (buffer r & 1), the barrier, wave 0 adds the 4 in wave
    // order and writes h (the next round's buffer is the other one: no second barrier)
    auto round = [&](WSet const &ws, int un, int r) {
      int const s = un / C::U13, u = un % C::U13;
      int const p0 = __builtin_amdgcn_readfirstlane(start[s]), n = __builtin_amdgcn_readfirstlane(start[s + 1]) - p0;
      int const tok = pl[p0 + (r16 < n ? r16 : n - 1)] / KP, c0 = wave * KBW;
      v4f tt2[2] = {v4f{0.f, 0.f, 0.f, 0.f}, v4f{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
      for (int kb = 0; kb < KBW; kb++) {
        v4f acc[2] = {v4f{0.f, 0.f, 0.f, 0.f}, v4f{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
        for (int st = 0; st < 4; st++) {
          v8bf const b = u4_bf16(*reinterpret_cast<uint4 const *>(xs + tok * XR + (c0 + kb) * 128 + 32 * grp + 8 * st));
#pragma unroll
          for (int i = 0; i < 2; i++) {
            uint4 const &w = ws.w[i][kb][st >> 1];
            v8bf const a = (st & 1) ? e4m3x8_bf16(w.z, w.w) : e4m3x8_bf16(w.x, w.y);
            acc[i] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, acc[i], 0, 0, 0);
          }
        }
        tt2[0] += acc[0] * ws.sg[kb];
        tt2[1] += acc[1] * ws.su[kb];
      }
      float *const pb = prs + (size_t)(r & 1) * (4 * 2 * 64 * 4);
#pragma unroll
      for (int i = 0; i < 2; i++) {
        *reinterpret_cast<v4f *>(pb + ((wave * 2 + i) * 64 + lane) * 4) = tt2[i];
      }
      __syncthreads();
      if (wave == 0) {
        v4f tot2[2];
#pragma unroll
        for (int i = 0; i < 2; i++) {
          tot2[i] = *reinterpret_cast<v4f const *>(pb + ((0 * 2 + i) * 64 + lane) * 4);
#pragma unroll
          for (int w = 1; w < 4; w++) {
            tot2[i] += *reinterpret_cast<v4f const *>(pb + ((w * 2 + i) * 64 + lane) * 4);
          }
        }
        if (r16 < n) {
          float hv[4];
#pragma unroll
          for (int q = 0; q < 4; q++) {
            float gv = tot2[0][q], uv = tot2[1][q];
            if (limit > 0.f) {
              gv = fminf(gv, limit);
              uv = fminf(fmaxf(uv, -limit), limit);
            }
            hv[q] = gv / (1.0f + __expf(-gv)) * uv;
          }
          int const p = pl[p0 + r16];
          uint2 const hw = make_uint2(pack_bf16x2(hv[0], hv[1]), pack_bf16x2(hv[2], hv[3]));
          __hip_atomic_store(reinterpret_cast<unsigned long long *>(hbuf + (size_t)p * IR + 16 * u + 4 * grp),
                             ((unsigned long long)hw.y << 32) | hw.x, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
      }
    };
    if (STATIC_GME_TIMES && lane == 0) {
      tms[1] = gtime();
    }
    int const nmain = n13u;
    constexpr int SHOFF = C::U13; // (slot 0's units done before the table)
    int un = cu + SHOFF, r = 0;
    if (un < nmain) {
      issue(wa, un);
    }
#pragma unroll 1
    while (un < nmain) {
      if (un + ncu < nmain) {
        issue(wb2, un + ncu);
      }
      __builtin_amdgcn_sched_barrier(0);
      round(wa, un, r);
      un += ncu;
      r++;
      if (un >= nmain) {
        break;
      }
      if (un + ncu < nmain) {
        issue(wa, un + ncu);
      }
      __builtin_amdgcn_sched_barrier(0);
      round(wb2, un, r);
      un += ncu;
      r++;
    }
    if (STATIC_GME_TIMES && lane == 0) {
      tms[2] = gtime();
    }
    // per slot its units done (slot s's count at ucnt[32 s]: its own 128-B line), counted by wave 0 once its h stores
    // completed: a W2 unit waits for its slot only (MI355X 11-07 glmmoe29, T 1: W2 started after every W13 unit, ~33 us,
    // the slots of the CUs with one round done at ~19)
    if (wave == 0) {
      wait_mem();
      if (lane == 0) {
        for (int v = cu + SHOFF; v < nmain; v += ncu) {
          __hip_atomic_fetch_add(ucnt + 32 * (v / C::U13), 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
      }
    }
    // this wave's first item of the generic loop (the W2 items): the first gw + k nw >= n13
    it0 = gw >= n13 ? gw : gw + ((n13 - gw + nw - 1) / nw) * nw;
  } else if constexpr (PIPE<C>) {
    // the pipelined W13 phase: x [T][H] into LDS (every CU the same rows), then this wave's W13 steps (item, chunk) in
    // order, step k + 1's weights (and block scales) loaded before step k's MFMAs (two register sets, the loop unrolled by
    // 2). MI355X 11-07 glmmoe25/26: a chunk at a time cost a round trip each (~5 us per 32 KB chunk)
    __hip_bfloat16 *const xs = reinterpret_cast<__hip_bfloat16 *>(sm + L::XS);
    constexpr int XR = H + 8;
    for (int v = tid; v < T * H / 8; v += 256) {
      int const t = v / (H / 8), c8 = 8 * (v % (H / 8));
      *reinterpret_cast<uint4 *>(xs + t * XR + c8) = ld16_in_at(x, (uint32_t)(((size_t)t * H + c8) * 2));
    }
    __syncthreads();
    constexpr int KBP = C::KB1 / KQ, CH = KBP < 8 ? KBP : 8, NCH = KBP / CH;
    static_assert(KBP % CH == 0, "glm_moe_experts: whole chunks");
    struct WSet {
      uint4 w[2][CH][2];
      float sg[CH], su[CH];
    };
    WSet wa, wb2;
    auto issue = [&](WSet &ws, int it, int c) {
      int const s = it / (C::U13 * KQ), u = (it / KQ) % C::U13, kq = it % KQ;
      int const e = __builtin_amdgcn_readfirstlane(se[s]);
      int const c0 = kq * KBP + c * CH;
      uint8_t const *wbp = w13 + ((size_t)e * C::U13 + u) * 32 * H;
#pragma unroll
      for (int i = 0; i < 2; i++)
#pragma unroll
        for (int kb = 0; kb < CH; kb++) {
          uint8_t const *rp = wbp + ((size_t)((c0 + kb) * 2 + i) * 2) * 1024 + 16 * lane;
          ws.w[i][kb][0] = *reinterpret_cast<uint4 const *>(rp);
          ws.w[i][kb][1] = *reinterpret_cast<uint4 const *>(rp + 1024);
        }
#pragma unroll
      for (int kb = 0; kb < CH; kb++) {
        size_t const so = (((size_t)e * C::IRB + u / 8) * C::KB1 + c0 + kb) * 2;
        ws.sg[kb] = w13sf[so];
        ws.su[kb] = w13sf[so + 1];
      }
    };
    v4f tot[2][NT];
    auto compute = [&](WSet const &ws, int it, int c) {
      int const s = it / (C::U13 * KQ), kq = it % KQ;
      int const p0 = __builtin_amdgcn_readfirstlane(start[s]), n = __builtin_amdgcn_readfirstlane(start[s + 1]) - p0;
      int const tok = pl[p0 + (r16 < n ? r16 : n - 1)] / KP;
      int const c0 = kq * KBP + c * CH;
      if (c == 0) {
        tot[0][0] = tot[1][0] = v4f{0.f, 0.f, 0.f, 0.f};
      }
#pragma unroll
      for (int kb = 0; kb < CH; kb++) {
        v4f acc[2] = {v4f{0.f, 0.f, 0.f, 0.f}, v4f{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
        for (int st = 0; st < 4; st++) {
          v8bf const b = u4_bf16(*reinterpret_cast<uint4 const *>(xs + tok * XR + (c0 + kb) * 128 + 32 * grp + 8 * st));
#pragma unroll
          for (int i = 0; i < 2; i++) {
            uint4 const &w = ws.w[i][kb][st >> 1];
            v8bf const a = (st & 1) ? e4m3x8_bf16(w.z, w.w) : e4m3x8_bf16(w.x, w.y);
            acc[i] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, acc[i], 0, 0, 0);
          }
        }
        tot[0][0] += acc[0] * ws.sg[kb];
        tot[1][0] += acc[1] * ws.su[kb];
      }
      if (c == NCH - 1) {
        if (STATIC_GME_TIMES && lane == 0 && it == gw) {
          tms[2] = gtime();
        }
        int const u = (it / KQ) % C::U13;
        finish13(tot, s, u, kq, p0, n, it);
      }
    };
    auto next = [&](int &it, int &c) {
      if (++c == NCH) {
        c = 0;
        it += nw;
      }
    };
    int it = gw, c = 0;
    if (it < n13) {
      issue(wa, it, c);
      if (STATIC_GME_TIMES && lane == 0) {
        tms[1] = gtime();
      }
    }
#pragma unroll 1
    while (it < n13) {
      int it2 = it, c2 = c;
      next(it2, c2);
      if (it2 < n13) {
        issue(wb2, it2, c2);
      }
      __builtin_amdgcn_sched_barrier(0);
      compute(wa, it, c);
      it = it2;
      c = c2;
      if (it >= n13) {
        break;
      }
      next(it2, c2);
      if (it2 < n13) {
        issue(wa, it2, c2);
      }
      __builtin_amdgcn_sched_barrier(0);
      compute(wb2, it, c);
      it = it2;
      c = c2;
    }
    it0 = it; // (the first of this wave's items past the W13 ones)
  }
  // BAL (generic path, SLOTW, DYP 0, KQ 1): the W2 items by weight - a W13 item reads W13W times a W2 item's bytes, the waves
  // with one more W13 item (gw < r13) take W13W fewer W2 items. Phase A: rounds 0 .. KA - 1 of the W2 items round robin over
  // every wave (j = gw + k nw); phase B: the rest round robin over the light waves only (j = KA nw + gw - r13 + m (nw - r13)).
  // MI355X 11-07 glmmoe56 (random T 64): waves 0 .. 159 (6 W13 + 5 W2 items) ended W2 at ~210-229 us, the others ~190-200
  constexpr int W13W = (32 * H / KQ + W2_ROWS * IR / 2) / (W2_ROWS * IR); // (T 64: 128 KB / 64 KB = 2)
  int const n2 = ntot - n13, r13 = n13 % nw;
  // (KA: the fewest rounds that leave at most W13W + 1 items per light wave for phase B - MI355X 11-07 glmab6: KA = (n2 + W13W
  // r13) / nw - W13W left up to ~47 items to each of the few light waves when r13 was close to nw: gmoe T 32 369.5 us)
  int const nl = nw - r13, rest = n2 - nl * (W13W + 1);
  int ka = rest <= 0 ? 0 : (rest + nw - 1) / nw;
  // (more rounds than whole ones: phase B over every wave, i.e. plain round robin)
  int const rb = ka > n2 / nw ? 0 : r13;
  ka = ka > n2 / nw ? n2 / nw : ka;
  bool const skipb = gw < rb;
  auto w2item = [&](int k) { // this wave's k-th W2 item (an item index, ntot: none)
    int const j = k < ka ? gw + k * nw : (skipb ? n2 : ka * nw + (gw - rb) + (k - ka) * (nw - rb));
    return j < n2 ? n13 + j : ntot;
  };
  // (KQ 1 only: MI355X 11-07 glmmoe60, random T 64 291.6 / 283.4 vs 294.9 us, T 32 211.3 vs 213.2; T 16 (KQ 2) 189.8 vs 167.4)
  constexpr int DYP = T >= 32 ? 2 : 0; // (the pooled W2 rounds; T 16: 0 = BAL, MI355X 11-08 glmmoe122 144.9 / 141.6 vs 148.8 / 149.8 us at 1)
  constexpr bool BAL = SLOTW<C> && DYP == 0 && KQ == 1;
  constexpr int W2P = W2_ROWS / 32;
  // (the loop's end without BAL: n13 + n2, as the whole rounds r2 plus the rest - this form keeps the compiled code of the
  // libraries measured with the W2 variants)
  int const r2 = n2 / nw * nw, rem2 = n2 - r2, ntv = n13 + r2 + rem2;
  int k2 = 0;
  if constexpr (BAL) {
    if (it0 >= n13) { // (no W13 item for this wave)
      it0 = w2item(k2++);
    }
  }
  // (DYP: items [n13 + n2s, ntot) from the counter)
  int const n2s = DYP ? (n2 > DYP * nw ? n2 - DYP * nw : 0) : n2;
  auto fetch = [&]() -> int {
    uint32_t idx = 0;
    if (lane == 0) {
      idx = __hip_atomic_fetch_add(cnt + 32, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    int const ii = __builtin_amdgcn_readfirstlane(__shfl((int)idx, 0));
    return n13 + n2s + ii < ntot ? n13 + n2s + ii : ntot;
  };
  auto next_it = [&](int it) -> int {
    if constexpr (BAL) {
      return it + nw < n13 ? it + nw : w2item(k2++);
    } else if constexpr (DYP > 0) {
      if (it + nw < n13 + n2s) {
        return it + nw;
      }
      return fetch();
    } else {
      return it + nw;
    }
  };
  if constexpr (DYP > 0) {
    if (it0 >= n13 + n2s) {
      it0 = fetch();
    }
  }
#pragma unroll 1
  for (int it = it0;; it = next_it(it)) {
    if (!w13_posted && it >= n13) {
      w13_posted = true;
      post_pend();
      wait_mem(); // (this wave's h stores done)
      if (STATIC_GME_TIMES && lane == 0) {
        tms[3] = gtime();
      }
      if (lane == 0) {
        if (atomicAdd(&misc[1], 1) == 3) {
          __hip_atomic_fetch_add(w13done, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
      }
    }
    if (it >= (BAL ? ntot : ntv)) {
      if (STATIC_GME_TIMES && lane == 0) { // (5 this wave's last item done, 6 its W2 items)
        tms[5] = gtime();
        tms[6] = n2done;
      }
      break;
    }
    if (STATIC_GME_DBG && lane == 0) {
      __hip_atomic_store(dbg + blockIdx.x, 4 + it, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    }
    if (STATIC_GME_TIMES && lane == 0 && it == it0 && n2done == 0) { // (7: this wave's first item of the generic loop entered)
      tms[7] = gtime();
    }
    bool const is13 = it < n13;
    int const jv = is13 ? 0 : it - n13; // (W2 item jv)
    // W13 item: (slot, unit, K piece kq) with the pieces of a unit adjacent
    int const s = is13 ? it / (C::U13 * KQ) : jv % ns, u = is13 ? (it / KQ) % C::U13 : jv / ns,
              kq = is13 ? it % KQ : 0;
    int const e = __builtin_amdgcn_readfirstlane(se[s]), p0 = __builtin_amdgcn_readfirstlane(start[s]),
              n = __builtin_amdgcn_readfirstlane(start[s + 1]) - p0;
    if (is13) {
      // W13 unit (this wave): rows 16 gate + 16 up of features 16 u .., K = H in chunks of 8 blocks of 128 (the chunk's
      // weight loads all issued first)
      uint8_t const *wb = w13 + ((size_t)e * C::U13 + u) * 32 * H;
      v4f tot[2][NT];
#pragma unroll
      for (int i = 0; i < 2; i++)
#pragma unroll
        for (int tt = 0; tt < NT; tt++) {
          tot[i][tt] = v4f{0.f, 0.f, 0.f, 0.f};
        }
      int tok[NT]; // (a column past the slot's tokens loads the slot's last token's row: its results are not stored)
#pragma unroll
      for (int tt = 0; tt < NT; tt++) {
        int const j = 16 * tt + r16;
        tok[tt] = pl[p0 + (j < n ? j : n - 1)] / KP;
      }
      // a chunk = CH K blocks of weights (32 KB per wave) issued at once, the x of the item's real token tiles NR (n tokens,
      // wave-uniform) in groups of XG blocks: the first group with the weights, the next one during this one's MFMAs.
      // MI355X 11-07 glmmoe36: with x per block (two register sets for NT tiles) every block after the first waited a
      // round trip for its x (loads complete in issue order, after the chunk's weights): ~32 round trips per T 64 item
      // (31 us per 128 KB, W13 at 2.8 TB/s). Most T 64 items have one tile (median 6 tokens per expert): x of the whole
      // chunk issued with its weights
      // (MI355X 11-07: without the sched_barrier the compiler interleaved the loads with the MFMAs and their waits, ~5-10
      // loads in flight per wave)
      constexpr int KBP = C::KB1 / KQ, CH = KBP < 8 ? KBP : 8;
      static_assert(KBP % CH == 0, "glm_moe_experts: whole chunks");
      auto body = [&](auto nr_c) {
        constexpr int NR = decltype(nr_c)::value;
        // (x registers XG x NR x 4 uint4 per buffer, two buffers when NG > 1: at most 32 uint4 = the old per-block double
        // buffer for NT 4. MI355X 11-07: groups of 4 / 2 blocks for NR 2 / 3+ spilled 185 VGPRs, T 64 e2e 295 vs 258 us)
        constexpr int XG0 = NR == 1 ? 8 : NR == 2 ? 2 : 1, XG = CH < XG0 ? CH : XG0, NG = CH / XG;
        static_assert(CH % XG == 0, "glm_moe_experts: whole x groups");
        auto ldx = [&](uint4 (&xr)[XG][NR][4], int kb0) {
#pragma unroll
          for (int j = 0; j < XG; j++)
#pragma unroll
            for (int tt = 0; tt < NR; tt++)
#pragma unroll
              for (int q = 0; q < 4; q++) {
                uint32_t const off = (uint32_t)(((size_t)tok[tt] * H + (kb0 + j) * 128 + 32 * grp + 8 * q) * 2);
                xr[j][tt][q] = ld16_in_at(x, off);
              }
        };
#pragma unroll 1
        for (int c0 = kq * KBP; c0 < (kq + 1) * KBP; c0 += CH) {
          uint4 wv[2][CH][2];
#pragma unroll
          for (int i = 0; i < 2; i++)
#pragma unroll
            for (int kb = 0; kb < CH; kb++) {
              uint8_t const *rp = wb + ((size_t)((c0 + kb) * 2 + i) * 2) * 1024 + 16 * lane; // (1 KB per wave load)
              wv[i][kb][0] = *reinterpret_cast<uint4 const *>(rp);
              wv[i][kb][1] = *reinterpret_cast<uint4 const *>(rp + 1024);
            }
          // the chunk's block scales with its weights (MI355X 11-07: loaded per block inside the MFMA loop after the
          // barrier, each scalar load's latency was waited for in turn)
          float sgv[CH], suv[CH];
#pragma unroll
          for (int kb = 0; kb < CH; kb++) {
            size_t const so = (((size_t)e * C::IRB + u / 8) * C::KB1 + c0 + kb) * 2;
            sgv[kb] = w13sf[so];
            suv[kb] = w13sf[so + 1];
          }
          uint4 xb[NG > 1 ? 2 : 1][XG][NR][4];
          ldx(xb[0], c0);
          __builtin_amdgcn_sched_barrier(0); // (every weight load of the chunk issued before any MFMA)
          if (STATIC_GME_TIMES && lane == 0 && it == gw && c0 == kq * KBP) {
            tms[1] = gtime();
          }
#pragma unroll
          for (int g = 0; g < NG; g++) {
            if (g + 1 < NG) {
              ldx(xb[(g + 1) & (NG > 1 ? 1 : 0)], c0 + (g + 1) * XG);
            }
#pragma unroll
            for (int j = 0; j < XG; j++) {
              int const kb = g * XG + j;
              auto const &xv = xb[g & (NG > 1 ? 1 : 0)][j];
              float const sg = sgv[kb], su = suv[kb];
              // (this lane's 32 K of the block: (c0 + kb) 128 + 32 grp, the same in A and B); the block's weights turned
              // into bf16 once for every token tile
              v8bf af[4][2];
#pragma unroll
              for (int st = 0; st < 4; st++)
#pragma unroll
                for (int i = 0; i < 2; i++) {
                  uint4 const &w = wv[i][kb][st >> 1];
                  af[st][i] = (st & 1) ? e4m3x8_bf16(w.z, w.w) : e4m3x8_bf16(w.x, w.y);
                }
#pragma unroll
              for (int tt = 0; tt < NR; tt++) {
                v4f acc[2] = {v4f{0.f, 0.f, 0.f, 0.f}, v4f{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
                for (int st = 0; st < 4; st++) {
                  v8bf const b = u4_bf16(xv[tt][st]);
#pragma unroll
                  for (int i = 0; i < 2; i++) {
                    acc[i] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(af[st][i], b, acc[i], 0, 0, 0);
                  }
                }
                tot[0][tt] += acc[0] * sg;
                tot[1][tt] += acc[1] * su;
              }
            }
          }
        }
      };
      // (the item's real tiles: n <= 16 NT tokens, wave-uniform; NR > 3 runs as NT)
      int const nr = (n + 15) >> 4;
      if (nr <= 1) {
        body(std::integral_constant<int, 1>{});
      } else if constexpr (NT >= 2) {
        if (nr == 2 || NT == 2) {
          body(std::integral_constant<int, 2>{});
        } else if constexpr (NT >= 3) {
          if (nr == 3 || NT == 3) {
            body(std::integral_constant<int, 3>{});
          } else {
            body(std::integral_constant<int, NT>{});
          }
        }
      }
      finish13(tot, s, u, kq, p0, n, it);
      // (the CU's count of its waves' W13 items: at the wave's first item past them)
    } else {
      // W2 unit (this wave): output rows W2_ROWS u .. (TL tiles of 16), K = IR (KB2 blocks of 128); the weights loaded per
      // tile pair
      constexpr int KB = C::KB2, TL = W2_ROWS / 16;
      float sc[KB];
#pragma unroll
      for (int kb = 0; kb < KB; kb++) {
        sc[kb] = w2sf[((size_t)e * C::U2 + (W2_ROWS * u) / 128) * KB + kb];
      }
      // the weights of a tile pair (1 KB per wave load); the first pair's before the wait for h (they do not depend on it).
      // P pairs per token tile, one register buffer (MI355X 11-07 glmmoe43, random x: two buffers, the next pair's weights
      // issued before this pair's MFMAs, T 64 350.4 / 357.4 vs 338.7 us, T 32 266.2 vs 250.9: slower)
      constexpr int P = TL / 2;
      static_assert(P == W2P, "glm_moe_experts: the W2 item's tile pairs");
      uint4 wv[1][2][KB][2];
      auto load_w2 = [&](uint4 (&wr)[2][KB][2], int t2) {
        uint8_t const *wb = w2 + ((size_t)e * C::U2 * 128 + W2_ROWS * u + 16 * t2) * IR; // (tile t2's 16 rows x IR)
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
          for (int kb = 0; kb < KB; kb++) {
            uint8_t const *rp = wb + (size_t)i * 16 * IR + (size_t)(kb * 2) * 1024 + 16 * lane;
            wr[i][kb][0] = *reinterpret_cast<uint4 const *>(rp);
            wr[i][kb][1] = *reinterpret_cast<uint4 const *>(rp + 1024);
          }
      };
      load_w2(wv[0], 0);
      if constexpr (CUW<C>) { // (its slot's W13 units)
        if (lane == 0) {
          while (cnt_ld(ucnt + 32 * s) < (uint32_t)C::U13) {
            __builtin_amdgcn_s_sleep(16);
          }
        }
        asm volatile("" ::: "memory");
      } else if constexpr (SLOTW<C>) { // (its slot's W13 units)
        if (lane == 0) {
          while (cnt_ld(ucnt + C::SC0 + 32 * s) < (uint32_t)C::U13) {
            __builtin_amdgcn_s_sleep(16);
          }
        }
        asm volatile("" ::: "memory");
      } else if (!w13_seen) {
        if (lane == 0) {
          if (atomicCAS(&misc[3], 0, 1) == 0) { // this CU's poller
            while (cnt_ld(w13done) < (uint32_t)ncu) {
              __builtin_amdgcn_s_sleep(127);
            }
            STAGE_STAMP_AT(1)
            __hip_atomic_store(&misc[2], 1, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP);
          } else {
            while (__hip_atomic_load(&misc[2], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_WORKGROUP) == 0) {
              __builtin_amdgcn_s_sleep(32);
            }
          }
        }
        asm volatile("" ::: "memory");
        w13_seen = true;
      }
      if (STATIC_GME_TIMES && lane == 0 && n2done == 0) { // (4 this wave's first W2 item past its wait)
        tms[4] = gtime();
      }
      n2done++;
      // token tiles outer: a tile's h loaded once for all TL / 2 tile pairs (the pairs' weights re-read from L2 for the
      // tiles after the first). MI355X 11-07 (real routing, T 64): h per (pair, tile) was a round trip each, W2 2.6 TB/s
#pragma unroll 1
      for (int tt = 0; tt < NT; tt++) {
        if (16 * tt >= n) {
          break;
        }
        int const j = 16 * tt + r16;
        int const p = j < n ? pl[p0 + j] : -1, pc = pl[p0 + (j < n ? j : n - 1)]; // (pc: past the tokens the last pair's h)
        uint4 hb[KB][4];
#pragma unroll
        for (int kb = 0; kb < KB; kb++)
#pragma unroll
          for (int q = 0; q < 4; q++) {
            uint32_t const off = (uint32_t)(((size_t)pc * IR + kb * 128 + 32 * grp + 8 * q) * 2);
            hb[kb][q] = ld16_in_at(hbuf, off);
          }
        float const wt = p >= 0 ? pw[p] : 0.f;
#pragma unroll
        for (int k = 0; k < P; k++) {
          int const t2 = 2 * k;
          if (tt > 0 || k > 0) { // (pair 0 of tile 0: loaded before the wait)
            load_w2(wv[0], t2);
          }
          auto const &wc = wv[0];
          __builtin_amdgcn_sched_barrier(0);
          v4f tot[2] = {v4f{0.f, 0.f, 0.f, 0.f}, v4f{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
          for (int kb = 0; kb < KB; kb++) {
            v4f acc[2] = {v4f{0.f, 0.f, 0.f, 0.f}, v4f{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
            for (int st = 0; st < 4; st++) {
              v8bf const b = u4_bf16(hb[kb][st]);
#pragma unroll
              for (int i = 0; i < 2; i++) {
                uint4 const &w = wc[i][kb][st >> 1];
                v8bf const a = (st & 1) ? e4m3x8_bf16(w.z, w.w) : e4m3x8_bf16(w.x, w.y);
                acc[i] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, acc[i], 0, 0, 0);
              }
            }
#pragma unroll
            for (int i = 0; i < 2; i++) {
              tot[i] += acc[i] * sc[kb];
            }
          }
          if (p >= 0) {
#pragma unroll
            for (int i = 0; i < 2; i++) {
              v4f const o = tot[i] * wt;
              st16_out(R + (size_t)p * H + W2_ROWS * u + 16 * (t2 + i) + 4 * grp,
                       make_uint4(__float_as_uint(o[0]), __float_as_uint(o[1]), __float_as_uint(o[2]),
                                  __float_as_uint(o[3])));
            }
          }
        }
      }
      { // (this item's R stores done, then its block counted)
        static_assert(128 % W2_ROWS == 0, "glm_moe_experts RBLK: whole items per 128-column block");
        wait_mem();
        if (lane == 0) {
          __hip_atomic_fetch_add(rblk + RBLK_STRIDE * (1 + (W2_ROWS * u) / 128), (uint32_t)(8 * W2_ROWS / 128), __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT);
        }
      }
    }
  }
}

template <class BUF_SLOTS>
__device__ __forceinline__ uint32_t *gme_rblk_ptr(G const &g) { // (R in blocks: buffer slot 9)
  return buf_at<uint32_t>(g, slot_at<BUF_SLOTS, 9>());
}
template <class BUF_SLOTS>
__device__ __forceinline__ int *gme_dbg_ptr(G const &g) {
  if constexpr (STATIC_GME_DBG || STATIC_GME_TIMES) {
    return buf_at<int>(g, slot_at<BUF_SLOTS, 6>());
  } else {
    return nullptr;
  }
}

// glm_moe_experts task x (one per CU). X: x's producer (hc_pre, or a graph input); PAIRS: glm_topk_route; PARAMS {NB (the
// bank's experts: routed + shared), KP (pairs per token), H, IR, swiglu limit (fp32 bits, 0: none), KQ (K pieces of a
// W13 unit)}; BUF_SLOTS {W13, W13 scales, W2, W2 scales, h (scratch), x (a graph input), dbg (or -1), the K pieces' sums
// (scratch, KQ > 1), the units' piece counts (scratch, 0 each launch, KQ > 1)}; its output R [T KP][H] fp32; counters: 0
// the tasks, 32 the queue, 64 the W13 units done
template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class X, class PAIRS, class... IN>
__device__ __forceinline__ void run_glm_moe_experts(Maps const &, G const &g, KernelLocals &L, StaticTask const &tk) {
  using C = gme::Cfg<PARAMS::v[0], PARAMS::v[1], PARAMS::v[2], PARAMS::v[3], PARAMS::v[5]>;
  static_assert(PAIRS::buf >= 0 && SELF::buf >= 0 && SELF::counter >= 0, "glm_moe_experts: the pairs, R, the counters");
  glm_moe_experts_task<C>(g,
                          buf_at<__hip_bfloat16 const>(g, X::buf >= 0 ? X::buf : slot_at<BUF_SLOTS, 5>()),
                          X::counter >= 0 ? cnt_of(g) + X::counter : nullptr,
                          (uint32_t)(X::x * X::y * X::z),
                          buf_at<unsigned long long const>(g, PAIRS::buf),
                          buf_at<uint8_t const>(g, slot_at<BUF_SLOTS, 0>()),
                          buf_at<float const>(g, slot_at<BUF_SLOTS, 1>()),
                          buf_at<uint8_t const>(g, slot_at<BUF_SLOTS, 2>()),
                          buf_at<float const>(g, slot_at<BUF_SLOTS, 3>()),
                          buf_at<__hip_bfloat16>(g, slot_at<BUF_SLOTS, 4>()),
                          C::KQ > 1 ? buf_at<float>(g, slot_at<BUF_SLOTS, 7>()) : nullptr,
                          C::KQ > 1 || gme::CUW<C> || gme::SLOTW<C> ? buf_at<uint32_t>(g, slot_at<BUF_SLOTS, 8>()) : nullptr,
                          buf_at<float>(g, SELF::buf),
                          cnt_of(g) + SELF::counter,
                          gme_rblk_ptr<BUF_SLOTS>(g),
                          __int_as_float(PARAMS::v[4]),
                          gme_dbg_ptr<BUF_SLOTS>(g),
                          tk.x,
                          SELF::x,
                          L.sm STAGE_STAMP_ARG(L.stage_stamps));
  task_post(cnt_of(g) + SELF::counter, 1);
  if (STATIC_GME_DBG && STATIC_TIDX == 0) {
    __hip_atomic_store(gme_dbg_ptr<BUF_SLOTS>(g) + blockIdx.x, 1000, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
  }
}

template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class... IN>
constexpr int smem_glm_moe_experts() {
  return gme::Lds<gme::Cfg<PARAMS::v[0], PARAMS::v[1], PARAMS::v[2], PARAMS::v[3], PARAMS::v[5]>>::BYTES;
}

} // namespace static_mk
