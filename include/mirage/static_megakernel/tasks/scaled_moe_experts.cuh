// tasks/scaled_moe_experts.cuh -- Compute the routed experts of an FP8 MoE.
// The checkpoint has e4m3 weights with one fp32 scale per 128 x 128 block.
// This task quantizes activations at run time. GLM-5.3 uses this format.
// scale_promotion.cuh defines the terms used below.
//   weights  e4m3, with the checkpoint's fp32 block scales (weight_scale_inv;
//            no requantization; mirage.mpk.fp8_fp32scale_weights)
//   x (z_q)  e4m3, one fp32 scale for each (token, 128 columns) (group_quant)
//   h (h_q)  e4m3, one fp32 scale for each (token, 128 features)
//   act      SiLU(gate) up (with a swiglu limit: gate clamped to at most the
//            limit and up to [-limit, limit] first, GLM-5.3-Flash)
// This task follows moe_experts.cuh, which uses MXFP4 weights and SiTU.
// scale_promotion.cuh supplies the FP8 MMA operations with fp32 scales.
// Both tasks use the same queue, warp roles, ring structure, and routing table.
// This task reuses the routing table functions and QueueEntry definition.
// Each SM takes entries from one queue (counter W2NEXT), in this order:
//   1. W13 items (expert slot, 64-feature tile m): the gate and up rows of the
//      expert x z_q (K = LAT).
//      a. Each piece has its own accumulator. The epilogue applies sw sx.
//      b. h = SiLU(gate) up.
//      c. h -> h_q. A group of 128 features is two items, one queue entry on
//         one SM: the second quantizes the group and adds 2 to CHQ[slot]
//         (h_quant_group_tok). The shared experts' slots come first. At most
//         NSM items (a small batch): item i on SM i instead, before its queue
//         entries, the two SMs of a group exchanging their amax
//         (h_quant_half_tok).
//   2. W2 entries contain two tiles. The last NSM tiles are in one-tile
//      entries.
//      A tile is (slot, 128-row output tile):
//      a. W2 x h_q (K = IR <= 256: K tile kt goes into accumulator half kt).
//         The epilogue applies sw sh.
//      b. For each token t of the expert: racc row (t, k) = weight(t, expert)
//         x result.
// The roles are as in moe_experts.cuh:
//   warp 0 lane 0  takes the entries and arms the even ring stages
//   warp 7         arms the odd ring stages
//   warps 1 and 6  issue the MMAs
//   warps 2..5     drain the accumulators
// TMEM columns: 0..63 and PROMO_COL1.. hold the accumulators of the pieces.
// CONST_SF holds the constant scale chunk (UE8M0 1 for both MMA operands).
// Sizes:
//   NE, K  the experts and the experts for each token (topk_route's params;
//          with sub-experts and shared experts, the bank's: NE DUP + SHARED
//          rounded up to whole 32-expert blocks (the routing table's; the
//          rounding's experts never chosen) and K DUP + SHARED)
//   LAT    z_q's width = the hidden size H (group_quant's params)
//   IR     the expert intermediate size on this GPU (the node's params[0])
//   LIMIT  the swiglu limit's fp32 bits (the node's params[1]; 0 or none: no
//          limit)
#pragma once
#include "../core.cuh"
#include "../scale_promotion.cuh" // the FP8 MMAs with fp32 scales
#include "group_quant.cuh"        // the layout of z_q (zq32_bytes)
#include "moe_experts.cuh" // QueueEntry, W13_TAKE_AT, table_poll, build_table
#include "topk_route.cuh"  // topk_dup

namespace static_mk {
namespace scaled_moe {

// what the bodies below use of the layer state, as slots and offsets
// (compile-time; run_scaled_moe_experts fills it from the node's slots):
// buffers HQ (h_q), HSF (its fp32 scales [NSLOT][T][KT2]), W13SF / W2SF (the
// fp32 weight scales, graph tensors); counters CNT (the node's first: SMs
// done), W2NEXT = CNT + 32 (the queue's next entry), CHQ = CNT + 64 .. (per
// expert slot the W13 items that wrote their h_q); tensor maps (Maps::m) of the
// expert weights (one and two pieces per box), of z_q (ZQ, ZQ_KT: set 0, set 1
// at + 1) and of h_q (HQ_KT); ZOFF: z_q's offset in the exchange region; the
// sizes NE, K, LAT, IR and the tile counts from them
template <int HQ_,
          int HSF_,
          int CNT_,
          int W13_,
          int W13X2_,
          int W13SF_,
          int W2_,
          int W2X2_,
          int W2SF_,
          int ZQ_,
          int ZQ_KT_,
          int HQ_KT_,
          size_t ZOFF_,
          int NE_,
          int K_,
          int LAT_,
          int IR_,
          int LIMIT_ = 0,
          int PROF_ = -1,
          int SHARED_ = 0,
          int LIVE_ = T>
struct Slots {
  static constexpr int HQ = HQ_, HSF = HSF_, CNT = CNT_, W2NEXT = CNT_ + 32,
                       CHQ = CNT_ + 64;
  static constexpr int W13 = W13_, W13X2 = W13X2_, W13SF = W13SF_, W2 = W2_,
                       W2X2 = W2X2_, W2SF = W2SF_;
  static constexpr int ZQ = ZQ_, ZQ_KT = ZQ_KT_, HQ_KT = HQ_KT_;
  static constexpr size_t ZOFF = ZOFF_;
  static constexpr int NE = NE_, K = K_, LAT = LAT_, IR = IR_, LIMIT = LIMIT_,
                       PROF = PROF_, SHARED = SHARED_; // SHARED: the shared
                                                       // bank experts (the
                                                       // last slots)
  static constexpr int NPAIR = T * K; // routing pairs of a step
  static constexpr int NSLOT =
      T * K < NE ? T * K : NE; // expert slots: at most every pair its own
                               // expert, at most every expert (T = 64)
  static constexpr int KT_LAT = LAT / 128; // K tiles of z_q (W13's K)
  static constexpr int MT13 =
      IR /
      64; // W13 items per expert: 64 features each (64 gate rows + 64 up rows)
  static constexpr int OT2 = LAT / 128; // W2 output tiles per expert
  static constexpr int KT2 = IR / 128;  // W2 K tiles
  static_assert(LAT % 128 == 0 && IR % 128 == 0,
                "scaled_moe_experts: whole 128-wide tiles");
  // single-item W13 entries (run_expert_dynamic): the layer of LIVE live rows
  // (topk_route's valid rows; the idle rows' pairs on the shared experts) has
  // at most LIVE (K - SHARED) + SHARED slots; when their W13 items are at most
  // NSM, each on its own SM (a small batch: bs 1 18 slots, 72 items). They use
  // the pair counters PAIRC .. (a slot's KT2 groups: NSM / 2) and the items'
  // amax ([slot][MT13][T] fp32, NSM T floats) at AMX floats into the h_q
  // scales, past those of the live slots (at most NSM T / 2 floats), so that
  // buffer must hold both (scaled_moe_experts_layer adds the counters under the
  // same condition)
  static constexpr int LIVE = LIVE_;
  static constexpr bool SINGLE = 2 * NSLOT * KT2 >= 3 * NSM &&
                                 (LIVE * (K - SHARED) + SHARED) * MT13 <= NSM;
  static constexpr int PAIRC = CHQ + NSLOT, AMX = NSM * T / 2;
  static_assert(NE % 32 == 0 && NE <= 1024,
                "scaled_moe_experts: build_table's 8 warps x 128 experts");
  // the routing table of this step, built on every SM by build_table (shared
  // memory, one per node)
  static __device__ __forceinline__ int *cnt_e() {
    __shared__ int a[NE];
    return a;
  } // per expert: > 0 if it has tokens, then its slot (build_table)
  static __device__ __forceinline__ int *sel() {
    __shared__ int a[NPAIR];
    return a;
  } // pair (t, k) -> expert
  static __device__ __forceinline__ float *wsel() {
    __shared__ float a[NPAIR];
    return a;
  } // pair (t, k) -> weight
  static __device__ __forceinline__ int *eid() {
    __shared__ int a[NSLOT];
    return a;
  } // per slot: expert id
  static __device__ __forceinline__ int *ntok() {
    __shared__ int a[NSLOT];
    return a;
  } // per slot: token count
  // the pairs grouped by slot (build_table): slot s's are entries start()[s]
  // .. + ntok()[s] - 1 of tok() (token), wt() (weight) and kk() (k: the
  // expert's place in the token's K)
  static __device__ __forceinline__ int *start() {
    __shared__ int a[NSLOT];
    return a;
  } // per slot: its first entry
  static __device__ __forceinline__ int *fill() {
    __shared__ int a[NSLOT];
    return a;
  } // per slot: entries placed so far (build_table)
  static __device__ __forceinline__ int *tok() {
    __shared__ int a[NPAIR];
    return a;
  } // per entry: token
  static __device__ __forceinline__ float *wt() {
    __shared__ float a[NPAIR];
    return a;
  } // per entry: weight
  static __device__ __forceinline__ unsigned char *kk() {
    __shared__ unsigned char a[NPAIR];
    return a;
  } // per entry: k
};

// the 128 epilogue threads: copies the weight scales of item (ex, m),
// [NATOMS][2], from the W13 scale tensor [E][IR / 128][NATOMS][2] (expert ex:
// GateUpScales) into bs. Then bar.sync 1, 128.
template <int NATOMS, int IR>
__device__ __forceinline__ void
    stage_w13_scales(float const *w13s, int ex, int m, float *bs) {
  float const *src = w13s + (size_t)ex * (IR / 128) * NATOMS * 2 +
                     GateUpScales::first(m, NATOMS, 0);
  for (int i = threadIdx.x - 64; i < NATOMS * GateUpScales::PER_PIECE;
       i += 128) {
    bs[i] = src[i];
  }
  asm volatile("bar.sync 1, 128;" ::: "memory");
}

// The two W13 items of a group of 128 features run on one SM, one after the
// other (one queue entry: W13_PAIR items), and the group is quantized there:
// exchanging h between two SMs (h and amax stores, an acq_rel add, the second
// item's loads) cost ~7 us an item at T = 32, where the epilogue's global
// accesses queue behind the weight stream and the issuers stall on the
// accumulator ticks meanwhile. The epilogues read and write only the columns
// of a slot's own tokens (the MMAs still run all T columns).
constexpr int W13_PAIR = 2;

// the 128 epilogue threads, after the drain of item mt of a slot
// into acc (128 rows x T fp32): column j of acc is token tok[j]'s (j < nt), or
// after a T-wide drain (wide) token j's (j < T: all tokens; the shared experts'
// slots have every token, a verify step's idle rows several times, nt > T; for
// another slot the tokens not its own get h_q rows W2 never reads). h of
// feature 64 mt + f is silu(acc[f][j], acc[64 + f][j]). Thread `row` takes
// feature f = row & 63 and the columns j = row >> 6, + 2, ... (all 128
// threads, half of the columns each). The loops run over the slot's tokens, not
// unrolled over T: this code runs once per item and misses the instruction
// cache each time (an unrolled T = 32 body is ~20 KB, fetched from L2 behind
// the weight stream: ~2 us a loop).
template <int LIMIT>
__device__ __forceinline__ float h_of(float const *acc, int f, int c) {
  return silu2_limit<LIMIT>(acc[f * T + c], acc[(64 + f) * T + c]);
}

// the group's first item: its h into ha ([64][T] fp32 in shared memory, column
// j); the next item's stage_w13_scales barrier orders it before the reads
template <int LIMIT>
__device__ __forceinline__ void h_first_tok(
    float *ha, float const *acc, int row, int const *tok, int nt, bool wide) {
  int const f = row & 63, n = wide ? T : nt;
#pragma unroll 1
  for (int j = row >> 6; j < n; j += 2) {
    ha[f * T + j] = h_of<LIMIT>(acc, f, j);
  }
}

// after the second item of the group kt2 (features 128 kt2 + 64 ..): per
// token the amax over the group's 128 features (ha and this item's h), the
// scale, the e4m3 values (h_q [NSLOT x T][IR], its scales hs [NSLOT][T][KT2]);
// then thread 64 adds 2 to CHQ[slot] (release; W2 waits for MT13). The other
// tokens' entries are not written (W2 reads none of them). H: acc holds this
// item's h ([64][T], T = 64's epilogue), not its gate and up rows.
template <int IR, int LIMIT, bool H = false>
__device__ __forceinline__ void h_quant_group_tok(uint8_t *hq,
                                                  float *hs,
                                                  uint32_t *chq,
                                                  int slot,
                                                  int kt2,
                                                  int row,
                                                  float const *acc,
                                                  float const *ha,
                                                  int const *tok,
                                                  int nt,
                                                  bool wide) {
  constexpr int KT2 = IR / 128;
  __shared__ float red_s[4 * T]; // [warp of the epilogue][token]
  int const lane = threadIdx.x & 31, f = row & 63, n = wide ? T : nt;
  auto hb = [&](int j) { // this item's h of feature f, column j
    if constexpr (H) {
      return acc[f * T + j];
    } else {
      return h_of<LIMIT>(acc, f, j);
    }
  };
#pragma unroll 1
  for (int j = row >> 6; j < n; j += 2) { // warp-uniform: row >> 6
    float am = fmaxf(fabsf(ha[f * T + j]), fabsf(hb(j)));
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
      am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, d));
    }
    if (lane == 0) {
      red_s[(row >> 5) * T + j] = am;
    }
  }
  asm volatile("bar.sync 1, 128;" ::: "memory");
#pragma unroll 1
  for (int j = row >> 6; j < n; j += 2) {
    int const t = wide ? j : tok[j], w0 = (row >> 6) * 2; // its two warps
    float const sc =
        fmaxf(fmaxf(red_s[w0 * T + j], red_s[(w0 + 1) * T + j]), 1.0e-10f) *
        (1.0f / 448.0f);
    uint8_t *const q = hq + ((size_t)slot * T + t) * IR + kt2 * 128 + f;
    q[0] = cvt_e4m3(__fdiv_rn(ha[f * T + j], sc));
    q[64] = cvt_e4m3(__fdiv_rn(hb(j), sc));
    if (f == 0) {
      hs[((size_t)slot * T + t) * KT2 + kt2] = sc;
    }
  }
  asm volatile("bar.sync 1, 128;" ::: "memory");
  if (threadIdx.x == 64) {
    cnt_add(chq + slot, 2); // release; the W2 loaders acquire MT13 for each
                            // slot
  }
}

// single-item W13 entries (every W13 item its SM's first entry, the two items
// of a group on two SMs): item mt quantizes its own 64 features of the group
// kt2 = mt / 2 with the group's scale. The two items exchange their per-token
// amax: amx [slot][MT13][T] fp32 stores, bar.sync, thread 64 adds 1 to the
// pair counter (cumulative release) and waits for 2 (acquire), bar.sync, the
// other item's amax by relaxed loads. Both items run on their SMs from the
// start, so the wait ends. The values are h_quant_group_tok's; item 2 kt2
// writes the scales. red: 4 T floats of shared memory. Then thread 64 adds 1
// to CHQ[slot].
template <int IR, int LIMIT>
__device__ __forceinline__ void h_quant_half_tok(uint8_t *hq,
                                                 float *hs,
                                                 float *amx,
                                                 uint32_t *pairc,
                                                 uint32_t *chq,
                                                 int slot,
                                                 int mt,
                                                 int row,
                                                 float const *acc,
                                                 float *red,
                                                 int const *tok,
                                                 int nt,
                                                 bool wide) {
  constexpr int KT2 = IR / 128, MT13 = IR / 64;
  int const lane = threadIdx.x & 31, f = row & 63, n = wide ? T : nt,
            w0 = (row >> 6) * 2, kt2 = mt >> 1;
#pragma unroll 1
  for (int j = row >> 6; j < n; j += 2) { // warp-uniform: row >> 6
    float am = fabsf(h_of<LIMIT>(acc, f, j));
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
      am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, d));
    }
    if (lane == 0) {
      red[(row >> 5) * T + j] = am;
    }
  }
  asm volatile("bar.sync 1, 128;" ::: "memory");
  float *const own = amx + ((size_t)slot * MT13 + mt) * T;
  if (f == 0) {
#pragma unroll 1
    for (int j = row >> 6; j < n; j += 2) {
      own[j] = fmaxf(red[w0 * T + j], red[(w0 + 1) * T + j]);
    }
  }
  asm volatile("bar.sync 1, 128;" ::: "memory");
  if (threadIdx.x == 64) {
    cnt_add(pairc + slot * KT2 + kt2, 1);
    while (cnt_ld(pairc + slot * KT2 + kt2) < 2) {
    }
  }
  asm volatile("bar.sync 1, 128;" ::: "memory");
  float const *const other = amx + ((size_t)slot * MT13 + (mt ^ 1)) * T;
#pragma unroll 1
  for (int j = row >> 6; j < n; j += 2) {
    int const t = wide ? j : tok[j];
    float const sc = fmaxf(fmaxf(fmaxf(red[w0 * T + j], red[(w0 + 1) * T + j]),
                                 ld_relaxed_f32(other + j)),
                           1.0e-10f) *
                     (1.0f / 448.0f);
    hq[((size_t)slot * T + t) * IR + mt * 64 + f] =
        cvt_e4m3(__fdiv_rn(h_of<LIMIT>(acc, f, j), sc));
    if (f == 0 && (mt & 1) == 0) {
      hs[((size_t)slot * T + t) * KT2 + kt2] = sc;
    }
  }
  asm volatile("bar.sync 1, 128;" ::: "memory");
  if (threadIdx.x == 64) {
    cnt_add(chq + slot, 1); // release; the W2 loaders acquire MT13 for each
                            // slot
  }
}

// issuer 0 (the elected lane) of a W2 tile: K tile kt goes into columns acc +
// 4 T kt (the tick's columns of issuer kt); its first MMA does not accumulate.
// sfc: the constant scale chunk. st: the W2 ring stage. HQ: the offset of h_q
// in the stage.
template <uint32_t IDESC, int KT2, int HQ>
__device__ __forceinline__ void
    issue_w2_promoted(uint32_t st, uint32_t acc, uint32_t sfc) {
  static_assert(KT2 <= 2, "scaled_moe_experts: W2's K tiles in the two halves");
#pragma unroll
  for (int kt = 0; kt < KT2; kt++) {
    uint32_t const d = acc + ACC_STAGES * T * kt;
    // h_q's K tile kt: T x 128 B after tile 0 (the TMA box {128, T, KT2}:
    // [KT2][T][128 B]), 8 T descriptor units
    uint64_t const dw = mkdesc(st + kt * 16384),
                   dh = mkdesc(st + HQ) + 8 * T * kt;
    if constexpr (F8_UNSCALED) { // T = 64 (runtime.cuh mma_f8)
      mma_f8(d, dw, dh, 0u);
      mma_f8(d, dw + 2, dh + 2, 1u);
      mma_f8(d, dw + 4, dh + 4, 1u);
      mma_f8(d, dw + 6, dh + 6, 1u);
    } else {
      mma_mx(d, dw, dh, idesc_mx(IDESC, 0), 0u, sfc, sfc);
      mma_mx(d, dw + 2, dh + 2, idesc_mx(IDESC, 1), 1u, sfc, sfc);
      mma_mx(d, dw + 4, dh + 4, idesc_mx(IDESC, 2), 1u, sfc, sfc);
      mma_mx(d, dw + 6, dh + 6, idesc_mx(IDESC, 3), 1u, sfc, sfc);
    }
  }
}

// issue_w2_promoted with h_q from a buffer at hq (T = 64: an entry's
// [KT2][T][128 B], w2_load_hq) instead of the stage
template <uint32_t IDESC, int KT2>
__device__ __forceinline__ void
    issue_w2_promoted_hq(uint32_t st, uint32_t hq, uint32_t acc, uint32_t sfc) {
  static_assert(KT2 <= 2, "scaled_moe_experts: W2's K tiles in the two halves");
#pragma unroll
  for (int kt = 0; kt < KT2; kt++) {
    uint32_t const d = acc + ACC_STAGES * T * kt;
    uint64_t const dw = mkdesc(st + kt * 16384), dh = mkdesc(hq) + 8 * T * kt;
    if constexpr (F8_UNSCALED) {
      mma_f8(d, dw, dh, 0u);
      mma_f8(d, dw + 2, dh + 2, 1u);
      mma_f8(d, dw + 4, dh + 4, 1u);
      mma_f8(d, dw + 6, dh + 6, 1u);
    } else {
      mma_mx(d, dw, dh, idesc_mx(IDESC, 0), 0u, sfc, sfc);
      mma_mx(d, dw + 2, dh + 2, idesc_mx(IDESC, 1), 1u, sfc, sfc);
      mma_mx(d, dw + 4, dh + 4, idesc_mx(IDESC, 2), 1u, sfc, sfc);
      mma_mx(d, dw + 6, dh + 6, idesc_mx(IDESC, 3), 1u, sfc, sfc);
    }
  }
}

// the 128 epilogue threads, after the acc_full of tick a: v[t] = the sum over
// kt of sw[kt] sh[kt][t] x half kt of the columns of the tick (the row of
// this thread)
template <int KT2>
__device__ __forceinline__ void drain_w2_promoted(
    uint32_t tb, int a, float const *sw, float const (*sh)[T], float *v) {
  uint32_t const lanes = (uint32_t)(((threadIdx.x >> 5) & 3) * 32) << 16;
  float p[T];
  // both halves' loads before one wait
  tmem_ld_cols_nw<T>(tb + T * a + lanes, v);
  if constexpr (KT2 == 2) {
    tmem_ld_cols_nw<T>(tb + T * a + ACC_STAGES * T + lanes, p);
  }
  tmem_keep_cols<T>(v);
  if constexpr (KT2 == 2) {
    tmem_keep_cols<T>(p);
  }
#pragma unroll
  for (int t = 0; t < T; t++) {
    v[t] *= sw[0] * sh[0][t];
  }
  if constexpr (KT2 == 2) {
#pragma unroll
    for (int t = 0; t < T; t++) {
      v[t] = fmaf(sw[1] * sh[1][t], p[t], v[t]);
    }
  }
}

// drain_w2_promoted with the h_q scales read from the staged buffer sbuf
// ([T][KT2] fp32, w2_stage_scales) instead of registers (T = 64: sh[2][T] in
// registers pushed the kernel past 255 registers)
template <int KT2>
__device__ __forceinline__ void drain_w2_promoted_s(
    uint32_t tb, int a, float const *sw, float const *sbuf, float *v) {
  uint32_t const lanes = (uint32_t)(((threadIdx.x >> 5) & 3) * 32) << 16;
  float p[T];
  tmem_ld_cols_nw<T>(tb + T * a + lanes, v);
  if constexpr (KT2 == 2) {
    tmem_ld_cols_nw<T>(tb + T * a + ACC_STAGES * T + lanes, p);
  }
  tmem_keep_cols<T>(v);
  if constexpr (KT2 == 2) {
    tmem_keep_cols<T>(p);
  }
#pragma unroll
  for (int t = 0; t < T; t++) {
    v[t] *= sw[0] * sbuf[t * KT2];
  }
  if constexpr (KT2 == 2) {
#pragma unroll
    for (int t = 0; t < T; t++) {
      v[t] = fmaf(sw[1] * sbuf[t * KT2 + 1], p[t], v[t]);
    }
  }
}

// ---- shared memory after the ring (offsets from the dynamic base) ----
// Keep the offsets from moe_experts.cuh's ExpertSmem layout.
// The areas for K3 scale chunks stay unused to preserve that layout.
// Do not remove this padding: the layer time changes with the position of
// these areas.
//   OFF_W2BAR   the W2 ring's barriers (w2_bar)
//   OFF_ACC  the accumulator of a W13 item (128 rows x T fp32, for SiLU across
//            rows)
//   OFF_BS   the weight scales of a W13 item ([KT_LAT][2])
//   OFF_XS   the scales of z_q ([KT_LAT][T])
//   OFF_HA   the first item's h of the group ([64][T] fp32,
//            h_quant_group_tok)
//   OFF_CSF  the phase switch stages the constant scale chunk here (the ring
//            area is idle at that time)
template <class RES>
struct Smem {
  static constexpr int OFF_CSF = OFF_W, OFF_W2BAR = RING_BYTES,
                       OFF_ACC =
                           OFF_W2BAR + RES::KT2 * 1024 + RES::KT2 * SF_CHUNK;
  static constexpr int OFF_BS = OFF_ACC + 128 * T * 4,
                       OFF_XS = OFF_BS + RES::KT_LAT * 2 * 4;
  static constexpr int OFF_HA = OFF_XS + RES::KT_LAT * T * 4;
  // T = 64: no OFF_HA (its 16 KB past the 227 KB with the routing tables):
  // OFF_ACC holds the two items' h instead ([64][T] each, the epilogue)
  static constexpr int BYTES =
      OFF_HA + (T == 64 ? 0 : 64 * T * 4); // the node's need from the
                                           // base
  static_assert(OFF_W2BAR % 1024 == 0, "128-B swizzled tiles are 1024-aligned");
  // ---- W2 ring: one tile per stage (KT2 x 16 KB weight pieces | h_q KT2 KB)
  // at the offsets of moe_experts.cuh's W2 ring. T = 64: the weights only (3
  // of 52 KB pass the 3-stage W13 ring's area, and 2 kept 2 tiles' weights in
  // flight), and an entry's h_q (its tiles' slot's: the same for both) in one
  // of two buffers W2_HQ0 + b W2_HQB after the stages (w2_load_hq) ----
  static constexpr int W2_ST = 3;
  static constexpr int W2_HQ =
      (RES::KT2 * 16384 + RES::KT2 * SF_CHUNK + 1023) / 1024 * 1024;
  static constexpr int W2_STB =
      T == 64 ? W2_HQ
              : (W2_HQ + RES::KT2 * T * 128 + RES::KT2 * SF_CHUNK + 4095) /
                    4096 * 4096;
  static constexpr int W2_HQB = RES::KT2 * T * 128, W2_HQ0 = W2_ST * W2_STB;
  static constexpr uint32_t W2_BYTES =
      RES::KT2 * 16384 + (T == 64 ? 0 : RES::KT2 * T * 128);
  static_assert(W2_ST * W2_STB + (T == 64 ? 2 * W2_HQB : 0) <= OFF_WSF,
                "the W2 ring lies inside the W13 ring's area");
  // the W2 entries' scales in the OFF_W2BAR area (w2_stage_scales): W2QD
  // buffers from OFF_W2S, one per queue position, each the slot's h_q scales
  // [T][KT2] then the 16-B chunk of the weight scales with the entry's tiles'
  static constexpr int W2S_HS = T * RES::KT2 * 4,
                       W2S_BYTES = (W2S_HS + 16 + 15) / 16 * 16,
                       OFF_W2S = OFF_W2BAR + (T == 64 ? 192 : 128);
  static_assert(OFF_W2S - OFF_W2BAR + W2QD * W2S_BYTES <=
                    RES::KT2 * 1024 + RES::KT2 * SF_CHUNK,
                "the W2 scale buffers fit the OFF_W2BAR area");
  // TMEM: the constant scale chunk at CONST_SF; the accumulators of piece 1
  // from PROMO_COL1. W2's K tiles go into the two accumulator halves.
  static constexpr int CONST_SF = 232;
  static_assert(RES::KT2 <= 2 && CONST_SF + 4 <= PROMO_COL1,
                "scaled_moe_experts: W2 K <= 256 (two accumulator halves)");
};
// mbarriers and one word in the (otherwise unused) OFF_W2BAR area: full [3],
// empty [3], ready (warp 0 has re-laid the ring), then warp 7's ring stage
// counter (W13 stages it has armed)
template <class RES>
__device__ __forceinline__ uint32_t w2_bar(uint32_t base, int i) {
  return base + Smem<RES>::OFF_W2BAR + 8 * i;
}
template <class RES>
__device__ __forceinline__ uint32_t w2_w7gl(uint32_t base) {
  return base + Smem<RES>::OFF_W2BAR + 64;
}
// the mbarrier of the scale buffer of queue position q (after the word above)
template <class RES>
__device__ __forceinline__ uint32_t w2s_bar(uint32_t base, int q) {
  return base + Smem<RES>::OFF_W2BAR + 72 + 8 * q;
}
// T = 64: the h_q buffers' mbarriers, full [2] then empty [2]
template <class RES>
__device__ __forceinline__ uint32_t w2hq_bar(uint32_t base, int i) {
  return base + Smem<RES>::OFF_W2BAR + 128 + 8 * i;
}

// The loader that arms the first tile of W2 entry (slot, ot0: the tile of the
// expert ex the entry starts at) at queue position q, after its w2_arm (this
// thread's CHQ acquire of the slot and its async-proxy fence came first, as
// for h_q): the entry's scales into buffer q by two bulk copies, completing on
// w2s_bar(q). The epilogue then reads them from shared memory: no acquire of
// its own and no global loads in its drains. A position is published again
// only after the epilogue retired its entry, so the buffer is free.
template <class RES>
__device__ __forceinline__ void w2_stage_scales(
    G const &g, uint32_t base, int q, int slot, int ex, int ot0) {
  using SM_ = Smem<RES>;
  constexpr int OT2 = RES::OT2, KT2 = RES::KT2;
  static_assert(KT2 == 2, "w2_stage_scales: 16 B of weight scales = 2 tiles");
  uint32_t const f = w2s_bar<RES>(base, q),
                 dst = base + SM_::OFF_W2S + q * SM_::W2S_BYTES;
  mbar_expect(f, SM_::W2S_HS + 16);
  bulk_g2s(dst,
           buf_at<float const>(g, RES::HSF) + (size_t)slot * T * KT2,
           SM_::W2S_HS,
           f);
  bulk_g2s(dst + SM_::W2S_HS,
           buf_at<float const>(g, RES::W2SF) +
               ((size_t)ex * OT2 + (ot0 & ~1)) * KT2,
           16,
           f);
}

// the box of tensor map d at (c0, c1, c2) into L2 (fire and forget)
__device__ __forceinline__ void
    tma3_prefetch_l2(void const *d, int c0, int c1, int c2) {
  asm volatile(
      "cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" ::
          "l"((uint64_t)d),
      "r"(c0),
      "r"(c1),
      "r"(c2)
      : "memory");
}

// ---- optional per-SM stamps: PROF >= 0 (the node's buffer slot 4, a graph
// tensor int64 [NSM][MOE_PROF_WORDS], scaled_moe_experts_layer(prof=)), in
// globaltimer ns; none of this is compiled without it. Per SM:
//   [0] run_expert_dynamic entered (the routing table is built)
//   [1] the epilogue saw the queue's end
//   [2] / [3] warp 0 starts / ends the switch to the W2 ring
//   [4] / [5] warp 0's W2 tiles: ns waiting for a free W2 stage / for the
//             slot's h_q (CHQ) in w2_arm
//   [6] / [7] the same for warp 7's W2 tiles
//   [9] / [10] / [11] thread 0: the task started / z_q, its scales and the
//             routing pairs landed (all threads') / the phase switch done
//   [12] thread 64: the phase switch done
//   [13] thread 0: past the task's first barrier (the SM's previous task's
//        warps done)
//   [14] / [15] thread 0: in build_table, past its first barrier / its third
//             (the slots numbered; the pairs' binary searches follow)
//   [8] epilogue events recorded, from [MOE_PROF_EV0]: kind << 56 | (time -
//       [0]); kind 1 a W13 item seen, 2 its h_q written; 3 a W2 tile seen, 4
//       its scales read (a new slot: also its CHQ wait and h_q scales), 5 its
//       accumulator full, 6 its rows stored
// The recording threads keep their counts, sums and [0] in registers and only
// store (no global load on the pipeline's path); the sums and the count at the
// queue's end.
constexpr int MOE_PROF_WORDS = 384, MOE_PROF_EV0 = 16;
__device__ __forceinline__ void
    moe_prof_event(long long *prof, int &n, long long t0, int kind) {
  if (MOE_PROF_EV0 + n < MOE_PROF_WORDS) {
    prof[MOE_PROF_EV0 + n] = ((long long)kind << 56) | (gtime() - t0);
    n++;
  }
}

// Arm W2 tile k of this SM:
//   1. Wait for its ring slot.
//   2. Load the weight pieces.
//   3. Wait until all W13 items of the tile's expert are done (the CHQ counter,
//      acquire, then the async-proxy fence).
//   4. Load its h_q. (The epilogue reads the h_q scales by pointer.)
// chk_slot: the slot that this loader checked last
template <class RES>
__device__ __forceinline__ void w2_arm(Maps const &maps,
                                       G const &g,
                                       uint32_t base,
                                       int k,
                                       int slot,
                                       int ex,
                                       int ot,
                                       int &chk_slot,
                                       long long &w_stage, // PROF: the waits
                                       long long &w_chq) {
  using SM_ = Smem<RES>;
  constexpr int W2_ST = SM_::W2_ST, W2_STB = SM_::W2_STB, W2_HQ = SM_::W2_HQ;
  constexpr int OT2 = RES::OT2, KT2 = RES::KT2;
  int const s = k % W2_ST;
  if (k >= W2_ST) {
    [[maybe_unused]] long long const t0 = RES::PROF >= 0 ? gtime() : 0;
    mbar_wait(w2_bar<RES>(base, 3 + s), ((k / W2_ST) - 1) & 1);
    if constexpr (RES::PROF >= 0) {
      w_stage += gtime() - t0;
    }
  }
  uint32_t const st = base + s * W2_STB, f = w2_bar<RES>(base, s);
  int const p0 = ex * (OT2 * KT2) + ot * KT2;
  mbar_expect(f, SM_::W2_BYTES);
  // the KT2 weight pieces: two per box, the last one alone when KT2 is odd (the
  // epilogue applies the fp32 weight scales)
#pragma unroll
  for (int kp = 0; kp + 1 < KT2; kp += 2) {
    tma3(&maps.m[RES::W2X2], f, st + kp * 16384, 0, 0, p0 + kp, EVICT_FIRST);
  }
  if constexpr (KT2 % 2 == 1) {
    tma3(&maps.m[RES::W2],
         f,
         st + (KT2 - 1) * 16384,
         0,
         0,
         p0 + KT2 - 1,
         EVICT_FIRST);
  }
  if (slot != chk_slot) {
    [[maybe_unused]] long long const t0 = RES::PROF >= 0 ? gtime() : 0;
    cnt_wait(cnt_now(g) + RES::CHQ + slot, RES::MT13);
    asm volatile("fence.proxy.async.global;" ::: "memory");
    chk_slot = slot;
    if constexpr (RES::PROF >= 0) {
      w_chq += gtime() - t0;
    }
  }
  if constexpr (T != 64) { // T = 64: the entry's h_q buffer (w2_load_hq)
    tma3(&maps.m[RES::HQ_KT], f, st + W2_HQ, 0, slot * T, 0, EVICT_LAST);
  }
}

// T = 64: W2 entry e2's h_q (its slot's [KT2][T][128 B]) into buffer e2 & 1,
// by the loader of its first tile after that tile's w2_arm (its CHQ acquire
// and async-proxy fence), once the issuer's MMAs of entry e2 - 2 are done
template <class RES>
__device__ __forceinline__ void
    w2_load_hq(Maps const &maps, uint32_t base, int e2, int slot) {
  using SM_ = Smem<RES>;
  int const b = e2 & 1;
  if (e2 >= 2) {
    mbar_wait(w2hq_bar<RES>(base, 2 + b), ((e2 >> 1) - 1) & 1);
  }
  uint32_t const f = w2hq_bar<RES>(base, b);
  mbar_expect(f, SM_::W2_HQB);
  tma3(&maps.m[RES::HQ_KT],
       f,
       base + SM_::W2_HQ0 + b * SM_::W2_HQB,
       0,
       slot * T,
       0,
       EVICT_LAST);
}

// all 8 warps of the SM, until the queue is empty. gl / gi0 / gi1: ring stage
// counters of the loaders / issuers; wt: accumulator ticks so far (the same on
// entry for every role). Timing build: stage stamps [0] the first entry's first
// stage landed at issuer 0 (the queue's first MMA), [1] issuer 0 takes its
// first W2 entry.
template <class RES>
__device__ __forceinline__ void
    run_expert_dynamic(G const &g,
                       Maps const &maps,
                       float *racc,
                       uint32_t base,
                       uint32_t tb,
                       int &gl,
                       int &gi0,
                       int &gi1,
                       int &wt,
                       int sm_id,
                       long long *prof STAGE_STAMP_PARAM) {
  using SM_ = Smem<RES>;
  constexpr int W2_ST = SM_::W2_ST, W2_STB = SM_::W2_STB, W2_HQ = SM_::W2_HQ;
  constexpr int KT_LAT = RES::KT_LAT, MT13 = RES::MT13, OT2 = RES::OT2,
                KT2 = RES::KT2, K = RES::K, LAT = RES::LAT, IR = RES::IR;
  int *const eid_s = RES::eid(), *const ntok_s = RES::ntok(),
             *const tok_s = RES::tok(), *const start_s = RES::start();
  float *const w_s = RES::wt();
  unsigned char *const kk_s = RES::kk();
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    for (int i = 0; i < W2_ST; i++) {
      mbar_init(w2_bar<RES>(base, i), 1);
      mbar_init(w2_bar<RES>(base, 3 + i), 1);
    }
    mbar_init(w2_bar<RES>(base, 6), 1);
    if constexpr (T == 64) {
      for (int i = 0; i < 4; i++) {
        mbar_init(w2hq_bar<RES>(base, i), 1);
      }
    }
    for (int q = 0; q < W2QD; q++) {
      mbar_init(w2s_bar<RES>(base, q), 1);
    }
    asm volatile("st.shared.u32 [%0], %1;" ::"r"(w2_w7gl<RES>(base)), "r"(gl)
                 : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  // W2 tiles n2 = slots x OT2: 2-tile entries, the last t1 = min(the tiles
  // after the first ~4/5 of the slots, NSM) in 1-tile entries, so the SMs
  // finish within about a tile of each other
  // W13 entries: W13_PAIR items each (a group of 128 features). A layer of at
  // most NSM items (RES::SINGLE, a small batch): one item each, item i SM i's
  // first entry (all SMs wait for the routing anyway), the queue only the W2
  // entries, a group quantized by its two SMs (h_quant_half_tok): two items on
  // one SM in turn took twice as long as the W13 phase needs (bs 1: 44.6 ->
  // 42.4 us a layer)
  static_assert(MT13 % W13_PAIR == 0, "scaled_moe_experts: whole groups");
  constexpr int npi = RES::SINGLE ? 1 : W13_PAIR;
  int const n_live = s_ints[1], n13s = RES::SINGLE ? n_live * MT13 : 0,
            n13 = RES::SINGLE ? 0 : n_live * (MT13 / W13_PAIR),
            n2 = n_live * OT2;
  int const n4 = ((n2 * 4 / 5) / OT2) * OT2,
            t1 = (n2 - n4 < NSM) ? n2 - n4 : NSM, off1 = n2 - t1,
            n2e = (off1 + 1) / 2;
  int const nentries = n13 + n2e + t1;
  // W2 tile q of the queue order: the shared experts' tiles first, as their
  // W13 items (all T tokens: the slowest tiles, not the last in the queue)
  int const w2rot = (n_live - RES::SHARED) * OT2;
  auto w2_tile = [&](int q) { // q + w2rot mod n2 (a multiple of OT2 apart)
    int const x = q + w2rot;
    return x >= n2 ? x - n2 : x;
  };
  float *acc_s = reinterpret_cast<float *>(rt_sm() + SM_::OFF_ACC);
  auto entry_of =
      [&](int idx, QueueEntry &e) { // entry number idx of the queue order above
        if (idx < n13) { // a: the slot, b: the entry's first item. The shared
                         // experts' slots (the last SHARED: all T tokens, the
                         // slowest items) first, then the others in order (as
                         // the W2 tiles that wait for them)
          e.kind = QE_W13;
          e.a = (idx / (MT13 / W13_PAIR) + n_live - RES::SHARED) % n_live;
          e.b = idx % (MT13 / W13_PAIR) * W13_PAIR;
        } else if (idx < n13 + n2e) {
          int const i0 = 2 * (idx - n13);
          e.kind = QE_W2;
          e.a = w2_tile(i0);
          e.b = (off1 - i0 < 2) ? off1 - i0 : 2;
        } else if (idx < nentries) {
          e.kind = QE_W2;
          e.a = w2_tile(off1 + (idx - n13 - n2e));
          e.b = 1;
        } else {
          e.kind = QE_END;
          e.a = e.b = 0;
        }
      };
  auto w13_single = [&](int i, QueueEntry &e) { // single: item i, in the
                                                // queue order's slot order
    e.kind = QE_W13;
    e.a = (i / MT13 + n_live - RES::SHARED) % n_live;
    e.b = i % MT13;
  };
  auto w13_job = [&](QueueEntry const &e,
                     int i,
                     TileJob &j) { // the ring job of item i of a W13 entry
    int const ex = eid_s[e.a];
    j = TileJob{};
    j.kind = 1;
    j.natoms = KT_LAT;
    j.nst = (KT_LAT + 1) / 2;
    j.k0 = 0;
    j.wmap = &maps.m[RES::W13];
    j.wmap2 = &maps.m[RES::W13X2]; // no scale maps: the epilogue applies them
    j.amap = &maps.m[RES::ZQ + dbuf_par];
    j.amap2 = &maps.m[RES::ZQ_KT + dbuf_par];
    j.tile0 = ex * (MT13 * KT_LAT) + (e.b + i) * KT_LAT;
  };
  if (warp == 0) {
    if (lane == 0) {
      int qn = 0, w2k = 0, e2 = 0,
          chk = -1; // w2k: this SM's W2 tiles so far; e2: its W2 entries
                    // (T = 64's h_q buffers); chk: the slot checked last
      bool switched = false, maps_pf = false;
      int idx = s_ints[2], pend = -1;
      [[maybe_unused]] long long pw_st = 0, pw_chq = 0; // PROF: the waits
      auto publish = [&](QueueEntry const &e,
                         int qn_) { // entry qn_ into both shared-memory copies
        int const qpos = qn_ % W2QD;
        if (qn_ >= W2QD) {
          mbar_wait(rt.qempty0 + 8 * qpos,
                    ((qn_ / W2QD) - 1) &
                        1); // the entry that used this position is retired
        }
        if (e.kind == QE_W2 &&
            !maps_pf) { // the W2 tensor maps -> the descriptor cache, one entry
                        // before the first W2 tile is armed
          prefetch_tmap(&maps.m[RES::W2X2]);
          prefetch_tmap(&maps.m[RES::W2]);
          prefetch_tmap(&maps.m[RES::HQ_KT]);
          maps_pf = true;
        }
        w2q[qpos][0] = e.kind;
        w2q[qpos][1] = e.a;
        w2q[qpos][2] = e.b;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbar_arrive(rt.qfull0 + 8 * qpos);
        if (qn_ >= W2QD) {
          mbar_wait(rt.qempty7_0 + 8 * qpos,
                    ((qn_ / W2QD) - 1) &
                        1); // warp 7 has read the previous entry here
        }
        w2q7[qpos][0] = e.kind;
        w2q7[qpos][1] = e.a;
        w2q7[qpos][2] = e.b;
        mbar_arrive(rt.qfull7_0 + 8 * qpos);
      };
      // W13 entries one ahead, the next taken once this item's first
      // W13_TAKE_AT stages are armed (so the SMs that arrive first take one
      // item each before any takes a second, as a launch gives every CTA one
      // tile); W2 entries two ahead
      QueueEntry e;
      entry_of(idx, e);
      if constexpr (RES::SINGLE) {
        if (sm_id < n13s) { // this SM's W13 item, then the queue from the
                            // entry taken at the start
          w13_single(sm_id, e);
          pend = idx;
        }
      }
      publish(e, qn);
      int n13_mine = 0;
      while (e.kind != QE_END) {
        QueueEntry en;
        if (e.kind == QE_W2) {
          int const idx1 =
              pend >= 0 ? pend : (int)atom_inc_lane0(cnt_now(g) + RES::W2NEXT);
          pend = (int)atom_inc_lane0(cnt_now(g) + RES::W2NEXT);
          if (!switched) {
            if constexpr (RES::PROF >= 0) {
              prof[2] = gtime();
            }
            // All stages of the W13 ring must be consumed before the W2 ring
            // writes over their area. The last use u of stage st0 (global
            // stage gq) completes `empty` phase u.
            // Warp 7 can still be arming its final stages.
            // Then use u - 1 can remain active, with the opposite parity.
            // A parity wait for u would pass before use u starts.
            // First wait for warp 7 to arm every stage below gl.
            // To arm use u, warp 7 must wait for use u - 1 to finish.
            // The barrier is then at phase u or u + 1, so the wait is valid.
            for (;;) {
              uint32_t g7;
              asm volatile("ld.volatile.shared.u32 %0, [%1];"
                           : "=r"(g7)
                           : "r"(w2_w7gl<RES>(base))
                           : "memory");
              if ((int)g7 >= gl) {
                break;
              }
              __nanosleep(32);
            }
            for (int st0 = 0; st0 < SMAX; st0++) {
              if (gl > st0) {
                int const gq = st0 + SMAX * ((gl - 1 - st0) / SMAX);
                mbar_wait(rt.empty0 + 8 * st0, (gq / SMAX) & 1);
              }
            }
            mbar_arrive(w2_bar<RES>(base, 6)); // warp 7 may load its W2 tiles
            switched = true;
            if constexpr (RES::PROF >= 0) {
              prof[3] = gtime();
            }
          }
          int const slot = e.a / OT2, ex = eid_s[slot], ot0 = e.a - slot * OT2;
          if ((w2k & 1) == 0) {
            scaled_moe::w2_arm<RES>(
                maps, g, base, w2k, slot, ex, ot0, chk, pw_st, pw_chq);
          }
          entry_of(idx1, en);
          publish(en, qn + 1);
          for (int i = 1; i < e.b; i++) {
            if (((w2k + i) & 1) == 0) {
              scaled_moe::w2_arm<RES>(maps,
                                      g,
                                      base,
                                      w2k + i,
                                      slot,
                                      ex,
                                      ot0 + i,
                                      chk,
                                      pw_st,
                                      pw_chq);
            }
          }
          if ((w2k & 1) == 0) { // this warp armed the entry's first tile
            if constexpr (T == 64) {
              scaled_moe::w2_load_hq<RES>(maps, base, e2, slot);
            }
            scaled_moe::w2_stage_scales<RES>(g, base, qn % W2QD, slot, ex, ot0);
          }
          w2k += e.b;
          e2++;
        } else {
          TileJob j;
          for (int i = 0; i + 1 < npi; i++) {
            w13_job(e, i, j);
            load_job<128, 16384, false>(j, base, gl, 0);
          }
          w13_job(e, npi - 1, j); // the next entry taken in the last item
          load_job<128, 16384, false>(j, base, gl, 0, 0, W13_TAKE_AT);
          n13_mine++;
          int idx1 = pend; // single: the entry taken at the start
          if (!RES::SINGLE || idx1 < 0) {
            idx1 = (int)atom_inc_lane0(cnt_now(g) + RES::W2NEXT);
          }
          pend = -1;
          load_job<128, 16384, false>(j, base, gl, 0, W13_TAKE_AT);
          entry_of(idx1, en);
          publish(en, qn + 1);
        }
        e = en;
        qn++;
      }
      if constexpr (RES::PROF >= 0) {
        prof[4] = pw_st;
        prof[5] = pw_chq;
      }
    }
  } else if (warp == 1 || warp == 6) { // the issuers: the whole warp runs the
                                       // loop, one elected lane issues
    int const iss = warp_same<true>((warp == 1) ? 0 : 1);
    int &gi = (warp == 1) ? gi0 : gi1;
    gi = warp_same<true>(gi);
    uint32_t const tbu = (uint32_t)warp_same<true>((int)tb),
                   baseu = (uint32_t)warp_same<true>((int)base);
    int wtt0 = warp_same<true>(wt), w2k = 0, e2 = 0; // e2: W2 entries (T = 64)
#ifdef STATIC_TIMING_BUILD
    bool w2_seen = false;
#endif
    for (int qn = 0;; qn++) {
      int const qpos = qn % W2QD;
      mbar_wait(rt.qfull0 + 8 * qpos, (qn / W2QD) & 1);
      QueueEntry const e{warp_same<true>(w2q[qpos][0]),
                         warp_same<true>(w2q[qpos][1]),
                         warp_same<true>(w2q[qpos][2])};
      if (e.kind == QE_END) {
        break;
      }
      if (e.kind == QE_W2) {
#ifdef STATIC_TIMING_BUILD
        if (stamp && iss == 0 && !w2_seen && issuer_lane<true>()) {
          stamp[1] = gtime();
        }
        w2_seen = true;
#endif
        if constexpr (T == 64) { // the entry's h_q landed (w2_load_hq)
          if (iss == 0) {
            mbar_wait(w2hq_bar<RES>(baseu, e2 & 1), (e2 >> 1) & 1);
          }
        }
        for (int i = 0; i < e.b;
             i++, w2k++) { // warp 1: the tile's 4 KT2 MMAs; warp 6: the
                           // second arrival on acc_full
          int const wtt = wtt0 + i, a = acc_slot(wtt), s = w2k % W2_ST;
          if (wtt >= ACC_STAGES) {
            mbar_wait(rt.acc_empty0 + 8 * a, (acc_round(wtt) - 1) & 1);
          }
          if (iss == 1) {
            if (issuer_lane<true>()) {
              mbar_arrive(rt.acc_full0 + 8 * a);
            }
            continue;
          }
          mbar_wait(w2_bar<RES>(baseu, s), (w2k / W2_ST) & 1);
          fence_after();
          if (issuer_lane<true>()) { // one K tile in each accumulator half
            if constexpr (T == 64) {
              issue_w2_promoted_hq<IDESC_MX_E4M3, KT2>(
                  baseu + s * W2_STB,
                  baseu + SM_::W2_HQ0 + (e2 & 1) * SM_::W2_HQB,
                  tbu + T * a,
                  tbu + SM_::CONST_SF);
            } else {
              issue_w2_promoted<IDESC_MX_E4M3, KT2, W2_HQ>(
                  baseu + s * W2_STB, tbu + T * a, tbu + SM_::CONST_SF);
            }
            tc_commit(w2_bar<RES>(baseu, 3 + s));
            tc_commit(rt.acc_full0 + 8 * a);
          }
        }
        if constexpr (T == 64) { // the h_q buffer free once these MMAs are done
          if (iss == 0 && issuer_lane<true>()) {
            tc_commit(w2hq_bar<RES>(baseu, 2 + (e2 & 1)));
          }
          e2++;
        }
        wtt0 += e.b;
      } else { // W13: 8 columns for each piece, PromoRounds ticks
               // (scale_promotion.cuh), for each item of the entry
        for (int i = 0; i < npi; i++) {
          issue_promoted<IDESC_MX_E4M3, KT_LAT>(
              baseu,
              tbu,
              iss,
              gi,
              wtt0,
              tbu + SM_::CONST_SF STAGE_STAMP_ARG(
                        (iss == 0 && qn == 0 && i == 0) ? stamp : nullptr));
          wtt0 += PromoRounds<KT_LAT>::ROUNDS;
        }
      }
    }
  } else if (warp == 7) { // second loader: the odd W13 stages, the odd W2 tiles
    int w2k7 = 0, chk7 = -1, e27 = 0;
    bool ready = false;
    [[maybe_unused]] long long pw_st7 = 0, pw_chq7 = 0; // PROF: the waits
    for (int qn = 0;; qn++) {
      int const qpos = qn % W2QD;
      mbar_wait(rt.qfull7_0 + 8 * qpos, (qn / W2QD) & 1);
      QueueEntry const e{w2q7[qpos][0], w2q7[qpos][1], w2q7[qpos][2]};
      __syncwarp();
      if (lane == 0) {
        mbar_arrive(rt.qempty7_0 + 8 * qpos);
      }
      if (e.kind == QE_END) {
        if constexpr (RES::PROF >= 0) {
          if (lane == 0) {
            prof[6] = pw_st7;
            prof[7] = pw_chq7;
          }
        }
        break;
      }
      if (e.kind == QE_W2) {
        if (!ready) {
          mbar_wait(w2_bar<RES>(base, 6), 0);
          ready = true;
        } // warp 0 has re-laid the ring
        if (lane == 0) {
          int const slot = e.a / OT2, ex = eid_s[slot], ot0 = e.a - slot * OT2;
          for (int i = 0; i < e.b; i++) {
            if (((w2k7 + i) & 1) == 1) {
              scaled_moe::w2_arm<RES>(maps,
                                      g,
                                      base,
                                      w2k7 + i,
                                      slot,
                                      ex,
                                      ot0 + i,
                                      chk7,
                                      pw_st7,
                                      pw_chq7);
            }
          }
          if ((w2k7 & 1) == 1) { // this warp armed the entry's first tile
            if constexpr (T == 64) {
              scaled_moe::w2_load_hq<RES>(maps, base, e27, slot);
            }
            scaled_moe::w2_stage_scales<RES>(g, base, qn % W2QD, slot, ex, ot0);
          }
        }
        w2k7 += e.b;
        e27++;
        __syncwarp();
      } else {
        for (int i = 0; i < npi; i++) {
          TileJob j;
          w13_job(e, i, j);
          if (lane == 0) {
            load_job<128, 16384, false>(j, base, gl, 1);
            asm volatile(
                "st.volatile.shared.u32 [%0], %1;" ::"r"(w2_w7gl<RES>(base)),
                "r"(gl)
                : "memory"); // for warp 0's switch
          } else {
            gl += j.nst;
          }
        }
      }
    }
  } else if (warp >= 2 &&
             warp <=
                 5) { // the epilogue, 128 threads (TMEM lane quadrant warp & 3)
    int wtt0 = wt;
    // the h_q scales of the entry's slot ([kt][t], its tokens' scale of each K
    // tile of W2), the same for all its tiles
    float sh[2][T];
    // the parities of the scale buffers' mbarriers (bit q: position q)
    uint32_t w2s_par = 0;
    [[maybe_unused]] int pn = 0;        // PROF: events so far (thread 64)
    [[maybe_unused]] long long pt0 = 0; // PROF: [0]
    if constexpr (RES::PROF >= 0) {
      if (threadIdx.x == 64) {
        pt0 = prof[0];
      }
    }
    for (int qn = 0;; qn++) {
      int const qpos = qn % W2QD;
      mbar_wait(rt.qfull0 + 8 * qpos, (qn / W2QD) & 1);
      QueueEntry const e{w2q[qpos][0], w2q[qpos][1], w2q[qpos][2]};
      [[maybe_unused]] bool const pev = RES::PROF >= 0 && threadIdx.x == 64;
      if (e.kind == QE_END) {
        if constexpr (RES::PROF >= 0) {
          if (pev) {
            prof[1] = gtime();
            prof[8] = pn;
          }
        }
        break;
      }
      if (e.kind == QE_W2) { // racc row (t, k) = weight(t) * result, for
                             // the slot's tokens (plain stores)
        // an entry's tiles are one slot's (2-tile entries start at even
        // tiles, OT2 is even)
        int const slot = e.a / OT2, nt = ntok_s[slot];
        int const lg = warp & 3, row = lg * 32 + lane;
        // the entry's scales from buffer qpos (w2_stage_scales: the loader of
        // its first tile staged them after its CHQ acquire): the weight scales
        // of its tiles ([ot - (ot0 & ~1)][KT2] in the 16-B chunk), the slot's
        // h_q scales
        mbar_wait(w2s_bar<RES>(base, qpos), (w2s_par >> qpos) & 1);
        w2s_par ^= 1u << qpos;
        float const *const sbuf = reinterpret_cast<float const *>(
            rt_sm() + SM_::OFF_W2S + qpos * SM_::W2S_BYTES);
        float const *const swc = sbuf + SM_::W2S_HS / 4;
        int const ot0 = e.a - slot * OT2;
        float sw[2][2];
#pragma unroll
        for (int item = 0; item < 2; item++) {
#pragma unroll
          for (int kt = 0; kt < 2; kt++) {
            sw[item][kt] = (item < e.b && kt < KT2)
                               ? swc[((ot0 & 1) + item) * KT2 + kt]
                               : 0.0f;
          }
        }
        if constexpr (T != 64) {
#pragma unroll
          for (int kt = 0; kt < 2; kt++) {
#pragma unroll
            for (int t = 0; t < T; t++) {
              sh[kt][t] = 0.0f;
              if (kt < KT2) {
                sh[kt][t] = sbuf[t * KT2 + kt];
              }
            }
          }
        }
        for (int item = 0; item < e.b; item++) {
          int const wtt = wtt0 + item, a = acc_slot(wtt),
                    par = acc_round(wtt) & 1, ot = e.a + item - slot * OT2;
          // sw[item] by a select (indexing with the run-time item put sw in
          // local memory)
          float const swi[2] = {item == 0 ? sw[0][0] : sw[1][0],
                                item == 0 ? sw[0][1] : sw[1][1]};
          float v[T];
          if constexpr (RES::PROF >= 0) {
            if (pev) {
              moe_prof_event(prof, pn, pt0, 3);
              moe_prof_event(prof, pn, pt0, 4);
            }
          }
          mbar_wait(rt.acc_full0 + 8 * a, par);
          if constexpr (RES::PROF >= 0) {
            if (pev) {
              moe_prof_event(prof, pn, pt0, 5);
            }
          }
          fence_after();
          if (T == 64 && nt <= 8) { // the slot's few tokens: their own
                                    // columns (one TMEM column of each K
                                    // half), not the T-wide drain
            uint32_t const lanes = (uint32_t)(((threadIdx.x >> 5) & 3) * 32)
                                   << 16,
                           c0a = tb + T * a + lanes, c1a = c0a + ACC_STAGES * T;
            int const s0 = start_s[slot];
            float c0[8], c1[8];
#pragma unroll
            for (int k = 0; k < 8; k++) {
              c0[k] = c1[k] = 0.f;
              if (k < nt) { // warp-uniform
                int const t = tok_s[s0 + k];
                c0[k] = tmem_ld1(c0a + t);
                if constexpr (KT2 == 2) {
                  c1[k] = tmem_ld1(c1a + t);
                }
              }
            }
            tmem_keep8(c0);
            tmem_keep8(c1);
            fence_before();
            mbar_arrive(rt.acc_empty0 + 8 * a);
#pragma unroll
            for (int k = 0; k < 8; k++) {
              if (k < nt) { // drain_w2_promoted's products, in its order
                int const i = s0 + k, t = tok_s[i];
                float vt = c0[k] * (swi[0] * sbuf[t * KT2]);
                if constexpr (KT2 == 2) {
                  vt = fmaf(swi[1] * sbuf[t * KT2 + 1], c1[k], vt);
                }
                racc[((size_t)t * K + kk_s[i]) * LAT + ot * 128 + row] =
                    w_s[i] * vt; // row (t, k), plain store
              }
            }
          } else {
            if constexpr (T == 64) {
              drain_w2_promoted_s<KT2>(tb, a, swi, sbuf, v);
            } else {
              drain_w2_promoted<KT2>(tb, a, swi, sh, v);
            }
            fence_before();
            mbar_arrive(rt.acc_empty0 + 8 * a);
            for (int i = start_s[slot]; i < start_s[slot] + nt; i++) {
              int const t = tok_s[i];
              float vt = v[0]; // v[t] by an unrolled select: indexing v with
                               // the run-time t put v in local memory (STL /
                               // LDL)
#pragma unroll
              for (int k = 1; k < T; k++) {
                vt = (t == k) ? v[k] : vt;
              }
              racc[((size_t)t * K + kk_s[i]) * LAT + ot * 128 + row] =
                  w_s[i] * vt; // row (t, k), plain store
            }
          }
          if constexpr (RES::PROF >= 0) {
            if (pev) {
              moe_prof_event(prof, pn, pt0, 6);
            }
          }
        }
        wtt0 += e.b;
      } else { // W13, each item of the entry: rows 0..63 gate, 64..127 up
               // of features [64 m, 64 m + 64) -> h_q of the slot
        int const slot = e.a;
        for (int i = 0; i < npi; i++) {
          int const mt = e.b + i;
          float v[T];
          if constexpr (RES::PROF >= 0) {
            if (pev) {
              moe_prof_event(prof, pn, pt0, 1);
            }
          }
          // the weight scales, then the rounds (the barrier in
          // stage_w13_scales also orders the previous item's acc_s reads
          // before this item's writes)
          float *const bs = reinterpret_cast<float *>(rt_sm() + SM_::OFF_BS);
          stage_w13_scales<KT_LAT, IR>(
              buf_at<float const>(g, RES::W13SF), eid_s[slot], mt, bs);
          if constexpr (RES::PROF >= 0) {
            if (pev) {
              moe_prof_event(prof, pn, pt0, 7);
            }
          }
          // the slot's tokens only
          int const nt = ntok_s[slot];
          int const *const tk = tok_s + start_s[slot];
          // more than 8 tokens (the shared experts' slots: all T): the T-wide
          // drain (the per-token one took ~60 us an item at nt = 32);
          // acc_s column t is token t's. Else column j is token tk[j]'s
          bool const wide = nt > 8;
          int row;
          if (wide) {
            row = drain_promoted<KT_LAT, GateUpScales>(
                tb,
                wtt0,
                bs,
                reinterpret_cast<float const *>(rt_sm() + SM_::OFF_XS),
                v);
          } else {
            row = drain_promoted_tok<KT_LAT, GateUpScales>(
                tb,
                wtt0,
                bs,
                reinterpret_cast<float const *>(rt_sm() + SM_::OFF_XS),
                tk,
                nt,
                v);
          }
          if constexpr (RES::PROF >= 0) {
            if (pev) {
              moe_prof_event(prof, pn, pt0, 8);
            }
          }
          if constexpr (T == 64) { // OFF_ACC: the first item's h [64][T], then
                                   // the up rows / the second item's h
            float *const h1 = acc_s, *const h2 = acc_s + 64 * T;
            if (row >= 64) {
#pragma unroll
              for (int j = 0; j < T; j++) {
                if (wide || j < nt) {
                  h2[(row - 64) * T + j] = v[j];
                }
              }
            }
            asm volatile("bar.sync 1, 128;" ::: "memory");
            if (row < 64) { // h of feature row: each element read, then
                            // written by its own thread
              float *const dst = i == 0 ? h1 : h2;
#pragma unroll
              for (int j = 0; j < T; j++) {
                if (wide || j < nt) {
                  dst[row * T + j] =
                      silu2_limit<RES::LIMIT>(v[j], h2[row * T + j]);
                }
              }
            }
            asm volatile("bar.sync 1, 128;" ::: "memory");
            if (i == 1) {
              h_quant_group_tok<IR, RES::LIMIT, true>(
                  buf_at<uint8_t>(g, RES::HQ),
                  buf_at<float>(g, RES::HSF),
                  cnt_now(g) + RES::CHQ,
                  slot,
                  mt >> 1,
                  row,
                  h2,
                  h1,
                  tk,
                  nt,
                  wide);
            }
          } else {
#pragma unroll
            for (int j = 0; j < T; j++) {
              if (wide || j < nt) {
                acc_s[row * T + j] = v[j];
              }
            }
            asm volatile("bar.sync 1, 128;" ::: "memory");
            float *const ha = reinterpret_cast<float *>(rt_sm() + SM_::OFF_HA);
            if constexpr (RES::SINGLE) { // the group's other item on another
                                         // SM
              h_quant_half_tok<IR, RES::LIMIT>(buf_at<uint8_t>(g, RES::HQ),
                                               buf_at<float>(g, RES::HSF),
                                               buf_at<float>(g, RES::HSF) +
                                                   RES::AMX,
                                               cnt_now(g) + RES::PAIRC,
                                               cnt_now(g) + RES::CHQ,
                                               slot,
                                               mt,
                                               row,
                                               acc_s,
                                               ha,
                                               tk,
                                               nt,
                                               wide);
            } else if (i == 0) { // the group's first item: its h for the
                                 // second
              h_first_tok<RES::LIMIT>(ha, acc_s, row, tk, nt, wide);
            } else {
              h_quant_group_tok<IR, RES::LIMIT>(buf_at<uint8_t>(g, RES::HQ),
                                                buf_at<float>(g, RES::HSF),
                                                cnt_now(g) + RES::CHQ,
                                                slot,
                                                mt >> 1,
                                                row,
                                                acc_s,
                                                ha,
                                                tk,
                                                nt,
                                                wide);
            }
          }
          if constexpr (RES::PROF >= 0) {
            if (pev) {
              moe_prof_event(prof, pn, pt0, 2);
            }
          }
          wtt0 += PromoRounds<KT_LAT>::ROUNDS;
        }
      }
      mbar_arrive(rt.qempty0 + 8 * qpos); // 128 arrivals: entry retired
    }
  }
}

// ---- the switch into the expert phase: its steps around the polls below
// (nothing in them waits for data, so they come first) ----
// begin: fresh ring barriers and stage counters, the constant scale chunk
// (UE8M0 1, for both MMA operands) into shared memory. Then, after a
// __syncthreads, copy_csf: warp 1 copies the chunk to TMEM column CONST_SF.
// end: wait for that copy.
template <class RES>
__device__ __forceinline__ void
    phase_switch_begin(uint32_t base, int &gl, int &gi0, int &gi1) {
  if (threadIdx.x == 0) {
    ring_reinit(false);
  }
  gl = gi0 = gi1 = 0;
  if (threadIdx.x < SF_CHUNK / 4) {
    const_sf_fill(base + Smem<RES>::OFF_CSF, threadIdx.x);
  }
}
template <class RES>
__device__ __forceinline__ void phase_switch_copy_csf(uint32_t base,
                                                      uint32_t tb) {
  if (threadIdx.x == 32) {
    if constexpr (!F8_UNSCALED) { // T = 64: no scale chunk (mma_f8)
      cp_sf(tb + Smem<RES>::CONST_SF, base + Smem<RES>::OFF_CSF);
    }
    tc_commit(rt.misc0 + 16);
  }
}
__device__ __forceinline__ void phase_switch_end() {
  mbar_wait(rt.misc0 + 16, 0);
  fence_after();
}

// wait until z_q (all K tiles) and its fp32 scales have landed (group_quant,
// this GPU's copy): thread 0 waits for zq_cnt (group_quant's counter: each of
// its zq_tasks tasks adds 1 after its stores), then the scales are staged into
// xs ([KT_LAT][T] fp32 in shared memory). Polling z_q's bytes instead (every SM
// its 128 KB at T = 32) spilled registers of the whole kernel and landed later.
// Then the 128 epilogue threads poll the NPAIR routing pairs (topk_route's
// buffer, 0xFF-prefilled) into the routing table's inputs (table_poll's steps:
// sel, wsel, cnt_e = 1), all of their pairs' loads per round. Before it:
// cnt_e zeroed, then a __syncthreads; after it: a __syncthreads before xs or
// the table are read.
template <class RES>
__device__ __forceinline__ void
    wait_z_and_pairs(G const &g,
                     unsigned long long const *pairs,
                     float *xs,
                     uint32_t const *zq_cnt,
                     uint32_t zq_tasks) {
  constexpr int NZV = (int)(zq32_bytes(RES::LAT) / 16), ZV = T * RES::LAT / 16,
                PP = (RES::NPAIR + 127) / 128;
  static_assert((NZV - ZV) * 4 == RES::KT_LAT * T,
                "z_q's fp32 scales [KT_LAT][T] follow its bytes (zq32_bytes)");
  if (threadIdx.x == 0) {
    cnt_wait(zq_cnt, zq_tasks);
  }
  __syncthreads();
  for (int piece = ZV + (int)threadIdx.x; piece < NZV; piece += 256) {
    uint4 const u =
        ld16_relaxed(g.rv + RES::ZOFF + (size_t)piece * 16 + rg_set());
    float *const d = xs + 4 * (piece - ZV);
    d[0] = __uint_as_float(u.x);
    d[1] = __uint_as_float(u.y);
    d[2] = __uint_as_float(u.z);
    d[3] = __uint_as_float(u.w);
  }
  int *const cntE_s = RES::cnt_e(), *const sel_ss = RES::sel();
  float *const wsel_ss = RES::wsel();
  int const pt = (int)threadIdx.x - 64; // an epilogue thread: pairs pt + 128 i
  bool pok[PP];
#pragma unroll
  for (int i = 0; i < PP; i++) {
    pok[i] = !(pt >= 0 && pt < 128 && pt + 128 * i < RES::NPAIR);
  }
  bool all;
  do {
    unsigned long long pv[PP];
#pragma unroll
    for (int i = 0; i < PP; i++) {
      if (!pok[i]) {
        asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];"
                     : "=l"(pv[i])
                     : "l"(pairs + pt + 128 * i)
                     : "memory");
      }
    }
    all = true;
#pragma unroll
    for (int i = 0; i < PP; i++) {
      if (!pok[i]) {
        pok[i] = pv[i] != ~0ull;
        if (pok[i]) {
          int const n = pt + 128 * i, e = (int)(pv[i] >> 32);
          sel_ss[n] = e;
          wsel_ss[n] = __uint_as_float((uint32_t)pv[i]);
          cntE_s[e] = 1; // the expert has a token
        }
      }
      all = all && pok[i];
    }
  } while (!all);
}

} // namespace scaled_moe

// the node's scaled_moe::Slots: its buffer slots (scaled_moe_experts_layer:
// h_q, its fp32 scales, the w13 scales (a tensor), the w2 scales (a tensor)),
// its map slots (w13, w13 x2, w2, w2 x2, z_q, z_q x2 K
// tiles (each 2 slots: the exchange region's sets), h_q), its counter, z_q's
// offset, the sizes: NE and K from topk_route's params (PAIRS; its bank:
// sub-experts and shared experts), LAT from group_quant's (ZQ), IR its own
// params[0], the swiglu limit its params[1] (if any)
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class ZQ,
          class PAIRS>
using ScaledMoeSlotsOf =
    scaled_moe::Slots<slot_at<BUF_SLOTS, 0>(),
                      slot_at<BUF_SLOTS, 1>(),
                      SELF::counter,
                      slot_at<MAP_SLOTS, 0>(),
                      slot_at<MAP_SLOTS, 1>(),
                      slot_at<BUF_SLOTS, 2>(),
                      slot_at<MAP_SLOTS, 2>(),
                      slot_at<MAP_SLOTS, 3>(),
                      slot_at<BUF_SLOTS, 3>(),
                      slot_at<MAP_SLOTS, 4>(),
                      slot_at<MAP_SLOTS, 5>(),
                      slot_at<MAP_SLOTS, 6>(),
                      exchange_offset[ZQ::buf],
                      (topk_bank_experts<PAIRS>() + 31) / 32 * 32,
                      topk_pairs_per_token<PAIRS>(),
                      ZQ::v[0],
                      PARAMS::v[0],
                      param_or<PARAMS, 1, 0>(),
                      slot_or<BUF_SLOTS, 4, -1>(),
                      topk_shared<PAIRS>(),
                      (param_or<PAIRS, 5, 0>() > 0 ? param_or<PAIRS, 5, 0>()
                                                   : T)>;

// scaled_moe_experts task (one per SM, task x on SM x): wait for z_q, phase
// switch, routing table, then the queue (above) -> the routed rows [T][K][LAT]
// in the node's output buffer; its counter counts the SMs done. IN: its
// inputs' producers, in the layer's order (z_q from group_quant, pairs from
// topk_route, ...). PARAMS {IR[, the swiglu limit as fp32 bits]}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class ZQ,
          class PAIRS,
          class... REST>
__device__ __forceinline__ void run_scaled_moe_experts(Maps const &maps,
                                                       G const &g,
                                                       KernelLocals &L,
                                                       StaticTask const &) {
  static_assert(
      SELF::buf >= 0 && SELF::counter >= 0 && PAIRS::buf >= 0,
      "scaled_moe_experts: its output rows, its counter, topk_route's pairs");
  using RES = ScaledMoeSlotsOf<SELF, PARAMS, BUF_SLOTS, MAP_SLOTS, ZQ, PAIRS>;
  static_assert(
      ZQ::buf >= 0 && exchange_bytes[ZQ::buf] == zq32_bytes(RES::LAT) &&
          ZQ::counter >= 0,
      "scaled_moe_experts: z_q in the exchange region (group_quant's, with "
      "its counter)");
  int const sm_id = L.sm_id;
  // PROF: each thread's own times of the task's start, the first barrier, z_q
  // landed, the pairs read (thread 0's and thread 64's are written below)
  [[maybe_unused]] long long pt_task = 0, pt_sync = 0, pt_z = 0, pt_pairs = 0;
  if constexpr (RES::PROF >= 0) {
    pt_task = gtime();
  }
  __syncthreads(); // every warp is done with the placed tiles (the loader runs
                   // ahead of the landings)
  if constexpr (RES::PROF >= 0) {
    pt_sync = gtime();
  }
  uint32_t pre_idx = 0;
  if (threadIdx.x == 0) {
    pre_idx = atom_inc_lane0(cnt_now(g) + RES::W2NEXT);
  }
  for (int i = threadIdx.x; i < RES::NE; i += 256) {
    RES::cnt_e()[i] = 0;
  }
  // the switch into the expert phase, its steps that wait for no data first;
  // then z_q, its scales and the routing pairs in one polling loop
  bool const relay = !L.ring_relaid;
  if (relay) {
    scaled_moe::phase_switch_begin<RES>(L.base, L.gl, L.gi0, L.gi1);
  }
  __syncthreads(); // cnt_e zeroed (the polls set it), the chunk staged
  if (relay) {
    scaled_moe::phase_switch_copy_csf<RES>(L.base, L.tb);
  }
  if constexpr (T == 64) {
    // most experts are chosen at T = 64: the first PFE experts' W13 weights
    // (the queue's first items) into L2 while the routing lands (HBM is idle
    // that long), each SM its share of their two-piece boxes
    constexpr int PFE = 40, B = RES::MT13 * RES::KT_LAT / 2;
    int const pi = sm_id + NSM * (threadIdx.x - 64);
    if (threadIdx.x >= 64 && threadIdx.x < 192 && pi < PFE * B) {
      scaled_moe::tma3_prefetch_l2(&maps.m[RES::W13X2],
                                   0,
                                   0,
                                   (pi / B) * (RES::MT13 * RES::KT_LAT) +
                                       2 * (pi % B));
    }
  }
  scaled_moe::wait_z_and_pairs<RES>(
      g,
      buf_at<unsigned long long const>(g, PAIRS::buf),
      reinterpret_cast<float *>(rt_sm() + scaled_moe::Smem<RES>::OFF_XS),
      cnt_now(g) + ZQ::counter,
      (uint32_t)(ZQ::x * ZQ::y));
  __syncthreads(); // every thread's pieces have landed and are staged
  // z_q came from generic stores (group_quant); the TMA loads read it next
  asm volatile("fence.proxy.async.global;" ::: "memory");
  if (threadIdx.x == 0) {
    g.stamps[sm_id * NSTAMP + STAMP_TASK0] = gtime();
  }
  if constexpr (RES::PROF >= 0) {
    pt_z = gtime();
  }
  if (relay) {
    scaled_moe::phase_switch_end();
    L.ring_relaid = true;
  }
  if constexpr (RES::PROF >= 0) {
    pt_pairs = gtime();
  }
  [[maybe_unused]] long long pt_bt[3] = {0, 0, 0}; // PROF: build_table's steps
  build_table<RES>(RES::PROF >= 0 ? pt_bt : nullptr);
  [[maybe_unused]] long long const pt_table = RES::PROF >= 0 ? gtime() : 0;
  long long *prof = nullptr;
  if constexpr (RES::PROF >= 0) {
    prof = buf_at<long long>(g, RES::PROF) +
           (size_t)sm_id * scaled_moe::MOE_PROF_WORDS;
  }
  if (threadIdx.x == 0) {
    s_ints[2] = (int)pre_idx;
    if constexpr (RES::PROF >= 0) {
      for (int i = 1; i < scaled_moe::MOE_PROF_EV0; i++) {
        prof[i] = 0;
      }
      prof[9] = pt_task;
      prof[10] = pt_z;
      prof[11] = pt_pairs;
      prof[13] = pt_sync;
      prof[14] = pt_bt[0];
      prof[15] = pt_bt[2];
      prof[0] = pt_table;
    }
  }
  __syncthreads();
  if constexpr (RES::PROF >= 0) {
    if (threadIdx.x == 64) { // a table_poll warp: its pairs landed
      prof[12] = pt_pairs;
    }
  }
  scaled_moe::run_expert_dynamic<RES>(g,
                                      maps,
                                      buf_at<float>(g, SELF::buf),
                                      L.base,
                                      L.tb,
                                      L.gl,
                                      L.gi0,
                                      L.gi1,
                                      L.wt,
                                      sm_id,
                                      prof STAGE_STAMP_ARG(L.stage_stamps));
  if (threadIdx.x == 0) {
    g.stamps[sm_id * NSTAMP + STAMP_TASK1] = gtime();
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    // this SM's routed rows are written (its consumer waits for all SMs)
    cnt_add(cnt_now(g) + SELF::counter, 1);
  }
}

// dynamic shared memory: the ring, then scaled_moe::Smem's
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class ZQ,
          class PAIRS,
          class... REST>
constexpr int smem_scaled_moe_experts() {
  return scaled_moe::Smem<
      ScaledMoeSlotsOf<SELF, PARAMS, BUF_SLOTS, MAP_SLOTS, ZQ, PAIRS>>::BYTES;
}

} // namespace static_mk
