// tasks/scaled_moe_experts.cuh -- Compute the routed experts of an FP8 MoE.
// The checkpoint has e4m3 weights with one fp32 scale per 128 x 128 block.
// This task quantizes activations at run time. GLM-5.3 uses this format.
// scale_promotion.cuh defines the terms used below.
//   weights  e4m3, with the checkpoint's fp32 block scales (weight_scale_inv;
//            no requantization; mirage.mpk.fp8_fp32scale_weights)
//   x (z_q)  e4m3, one fp32 scale for each (token, 128 columns) (group_quant)
//   h (h_q)  e4m3, one fp32 scale for each (token, 128 features)
//   act      SiLU(gate) up
// This task follows moe_experts.cuh, which uses MXFP4 weights and SiTU.
// scale_promotion.cuh supplies the FP8 MMA operations with fp32 scales.
// Both tasks use the same queue, warp roles, ring structure, and routing table.
// This task reuses the routing table functions and QueueEntry definition.
// Each SM takes entries from one queue (counter W2NEXT), in this order:
//   1. W13 items (expert slot, 64-feature tile m): the gate and up rows of the
//      expert x z_q (K = LAT).
//      a. Each piece has its own accumulator. The epilogue applies sw sx.
//      b. h = SiLU(gate) up.
//      c. h -> h_q. A group of 128 features is two items, maybe on two SMs.
//         The item that finishes second quantizes the group and adds 2 to
//         CHQ[slot] (h_quant_pair).
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
//   NE, K  the experts and the experts for each token (topk_route's params)
//   LAT    z_q's width = the hidden size H (group_quant's params)
//   IR     the expert intermediate size on this GPU (the node's params[0])
#pragma once
#include "../core.cuh"
#include "../scale_promotion.cuh" // the FP8 MMAs with fp32 scales
#include "group_quant.cuh"        // the layout of z_q (zq32_bytes)
#include "moe_experts.cuh" // QueueEntry, W13_TAKE_AT, table_poll, build_table

namespace static_mk {
namespace scaled_moe {

// what the bodies below use of the layer state, as slots and offsets
// (compile-time; run_scaled_moe_experts fills it from the node's slots):
// buffers HQ (h_q), HSF (its fp32 scales [NSLOT][T][KT2]), H32 (the scratch of
// h, fp32, and its amax: h_quant_pair), W13SF / W2SF (the fp32 weight scales,
// graph tensors); counters CNT (the node's first: SMs done), W2NEXT = CNT + 32
// (the queue's next entry), CHQ = CNT + 64 .. (per expert slot the W13 items
// that wrote their h_q), PAIRC = CHQ + NSLOT .. (per slot and 128 h features
// the items done); tensor maps (Maps::m) of the expert weights (one and two
// pieces per box), of z_q (ZQ, ZQ_KT: set 0, set 1 at + 1) and of h_q (HQ_KT);
// ZOFF: z_q's offset in the exchange region; the sizes NE, K, LAT, IR and the
// tile counts from them
template <int HQ_,
          int HSF_,
          int CNT_,
          int W13_,
          int W13X2_,
          int W13SF_,
          int H32_,
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
          int IR_>
struct Slots {
  static constexpr int HQ = HQ_, HSF = HSF_, H32 = H32_, CNT = CNT_,
                       W2NEXT = CNT_ + 32, CHQ = CNT_ + 64;
  static constexpr int W13 = W13_, W13X2 = W13X2_, W13SF = W13SF_, W2 = W2_,
                       W2X2 = W2X2_, W2SF = W2SF_;
  static constexpr int ZQ = ZQ_, ZQ_KT = ZQ_KT_, HQ_KT = HQ_KT_;
  static constexpr size_t ZOFF = ZOFF_;
  static constexpr int NE = NE_, K = K_, LAT = LAT_, IR = IR_;
  static constexpr int NPAIR = T * K; // routing pairs of a step
  static constexpr int NSLOT =
      T * K; // expert slots: at most every pair its own expert
  static constexpr int PAIRC = CHQ + NSLOT;
  static constexpr int KT_LAT = LAT / 128; // K tiles of z_q (W13's K)
  static constexpr int MT13 =
      IR /
      64; // W13 items per expert: 64 features each (64 gate rows + 64 up rows)
  static constexpr int OT2 = LAT / 128; // W2 output tiles per expert
  static constexpr int KT2 = IR / 128;  // W2 K tiles
  static_assert(LAT % 128 == 0 && IR % 128 == 0,
                "scaled_moe_experts: whole 128-wide tiles");
  static_assert(NE % 32 == 0 && NE <= 1024 && NPAIR <= 256,
                "scaled_moe_experts: build_table's 8 warps x 128 experts, a "
                "thread per pair");
  // the routing table of this step, built on every SM by build_table (shared
  // memory, one per node)
  static __device__ __forceinline__ int *cnt_e() {
    __shared__ int a[NE];
    return a;
  } // tokens per expert
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
  static __device__ __forceinline__ int *tok() {
    __shared__ int a[NSLOT * T];
    return a;
  } // per slot: tokens
  static __device__ __forceinline__ float *wt() {
    __shared__ float a[NSLOT * T];
    return a;
  } // per slot and token: weight
  // k: the expert's place in the token's K
  static __device__ __forceinline__ unsigned char *kk() {
    __shared__ unsigned char a[NSLOT * T];
    return a;
  }
};

__device__ __forceinline__ uint32_t atom_add_acq_rel(uint32_t *c, uint32_t v) {
  uint32_t old;
  asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
               : "=r"(old)
               : "l"(c), "r"(v)
               : "memory");
  return old;
}

// copies the fp32 scales of z_q ([NKT][T], group_quant) into xs in shared
// memory. All 256 threads call it after wait_z_landed and a __syncthreads (a
// thread polls only its own pieces of z_q). A __syncthreads must come before
// the first use of xs.
template <int NKT>
__device__ __forceinline__ void stage_x_scales(unsigned char const *src,
                                               float *xs) {
  for (int i = threadIdx.x; i < NKT * T; i += blockDim.x) {
    xs[i] = ld_relaxed_f32(reinterpret_cast<float const *>(src) + i);
  }
}

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

// the 128 epilogue threads, after W13 item (slot, mt). The threads of rows <
// 64 hold h[t] of feature 64 mt + row. The item writes its part of h32
// [NSLOT][T][IR] and of amax [NSLOT][MT13][T] (fp32, the scratch of the
// node). Counter pairc[slot][mt / 2] (zero before each launch) counts the
// items of the group of 128 features. The item that finishes second writes
// the h_q of the group ([NSLOT x T][IR] e4m3) and its scales (hs
// [NSLOT][T][KT2]). Then it adds 2 to CHQ[slot] (release; W2 waits for MT13).
// Memory order: all threads store, then bar.sync, then thread 64 does the
// acq_rel add (a cumulative release; for the second item also the acquire).
// After the next bar.sync, the second item reads with gpu-scope relaxed loads.
template <int IR>
__device__ __forceinline__ void h_quant_pair(float *h32,
                                             float *amax,
                                             uint8_t *hq,
                                             float *hs,
                                             uint32_t *pairc,
                                             uint32_t *chq,
                                             int slot,
                                             int mt,
                                             int row,
                                             float const *h) {
  constexpr int MT13 = IR / 64, KT2 = IR / 128;
  __shared__ float red_s[2 * T];
  __shared__ int second_s;
  int const lane = threadIdx.x & 31;
  if (row < 64) {
    int const f = mt * 64 + row;
    float am[T];
#pragma unroll
    for (int t = 0; t < T; t++) {
      h32[((size_t)slot * T + t) * IR + f] = h[t];
      am[t] = fabsf(h[t]);
    }
#pragma unroll
    for (int d = 16; d > 0; d >>= 1)
#pragma unroll
      for (int t = 0; t < T; t++) {
        am[t] = fmaxf(am[t], __shfl_xor_sync(0xffffffffu, am[t], d));
      }
    if (lane == 0) {
#pragma unroll
      for (int t = 0; t < T; t++) {
        red_s[(row >> 5) * T + t] = am[t];
      }
    }
  }
  asm volatile("bar.sync 1, 128;" ::: "memory");
  if (threadIdx.x == 64) {
#pragma unroll
    for (int t = 0; t < T; t++) {
      amax[((size_t)slot * MT13 + mt) * T + t] = fmaxf(red_s[t], red_s[T + t]);
    }
    second_s = atom_add_acq_rel(pairc + slot * KT2 + (mt >> 1), 1u) == 1u;
  }
  asm volatile("bar.sync 1, 128;" ::: "memory");
  if (second_s) { // the 128 features of the group, one for each thread
    int const kt2 = mt >> 1, f = kt2 * 128 + row;
    float s[T];
#pragma unroll
    for (int t = 0; t < T; t++) {
      float const a0 = ld_relaxed_f32(amax +
                                      ((size_t)slot * MT13 + 2 * kt2) * T + t),
                  a1 = ld_relaxed_f32(
                      amax + ((size_t)slot * MT13 + 2 * kt2 + 1) * T + t);
      s[t] = fmaxf(fmaxf(a0, a1), 1.0e-10f) * (1.0f / 448.0f);
      float const v = ld_relaxed_f32(h32 + ((size_t)slot * T + t) * IR + f);
      hq[((size_t)slot * T + t) * IR + f] = cvt_e4m3(__fdiv_rn(v, s[t]));
    }
    if (row == 0) {
#pragma unroll
      for (int t = 0; t < T; t++) {
        hs[((size_t)slot * T + t) * KT2 + kt2] = s[t];
      }
    }
    asm volatile("bar.sync 1, 128;" ::: "memory");
    if (threadIdx.x == 64) {
      cnt_add(chq + slot, 2); // release; the W2 loaders acquire MT13 for each
                              // slot
    }
  }
}

// issuer 0 (the elected lane) of a W2 tile: K tile kt goes into columns acc +
// 32 kt; its first MMA does not accumulate. sfc: the constant scale chunk. st:
// the W2 ring stage. HQ: the offset of h_q in the stage.
template <uint32_t IDESC, int KT2, int HQ>
__device__ __forceinline__ void
    issue_w2_promoted(uint32_t st, uint32_t acc, uint32_t sfc) {
  static_assert(KT2 <= 2, "scaled_moe_experts: W2's K tiles in the two halves");
#pragma unroll
  for (int kt = 0; kt < KT2; kt++) {
    uint32_t const d = acc + 32 * kt;
    uint64_t const dw = mkdesc(st + kt * 16384), dh = mkdesc(st + HQ) + 64 * kt;
    mma_mx(d, dw, dh, idesc_mx(IDESC, 0), 0u, sfc, sfc);
    mma_mx(d, dw + 2, dh + 2, idesc_mx(IDESC, 1), 1u, sfc, sfc);
    mma_mx(d, dw + 4, dh + 4, idesc_mx(IDESC, 2), 1u, sfc, sfc);
    mma_mx(d, dw + 6, dh + 6, idesc_mx(IDESC, 3), 1u, sfc, sfc);
  }
}

// the 128 epilogue threads, after the acc_full of tick a: v[t] = the sum over
// kt of sw[kt] sh[kt][t] x half kt of the columns of the tick (the row of
// this thread)
template <int KT2>
__device__ __forceinline__ void drain_w2_promoted(
    uint32_t tb, int a, float const *sw, float const (*sh)[T], float *v) {
  uint32_t const lanes = (uint32_t)(((threadIdx.x >> 5) & 3) * 32) << 16;
  tmem_ld8(tb + 8 * a + lanes, v);
#pragma unroll
  for (int t = 0; t < 8; t++) {
    v[t] *= sw[0] * sh[0][t];
  }
  if constexpr (KT2 == 2) {
    float p[8];
    tmem_ld8(tb + 8 * a + 32 + lanes, p);
#pragma unroll
    for (int t = 0; t < 8; t++) {
      v[t] = fmaf(sw[1] * sh[1][t], p[t], v[t]);
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
//   OFF_CSF  the phase switch stages the constant scale chunk here (the ring
//            area is idle at that time)
template <class RES>
struct Smem {
  static constexpr int OFF_CSF = OFF_W, OFF_W2BAR = RING_BYTES,
                       OFF_ACC =
                           OFF_W2BAR + RES::KT2 * 1024 + RES::KT2 * SF_CHUNK;
  static constexpr int OFF_BS = OFF_ACC + 128 * T * 4,
                       OFF_XS = OFF_BS + RES::KT_LAT * 2 * 4;
  static constexpr int BYTES =
      OFF_XS + RES::KT_LAT * T * 4; // the node's need from the base
  static_assert(OFF_W2BAR % 1024 == 0, "128-B swizzled tiles are 1024-aligned");
  // ---- W2 ring: one tile per stage (KT2 x 16 KB weight pieces | h_q KT2 KB)
  // at the offsets of moe_experts.cuh's W2 ring ----
  static constexpr int W2_ST = 3;
  static constexpr int W2_HQ =
      (RES::KT2 * 16384 + RES::KT2 * SF_CHUNK + 1023) / 1024 * 1024;
  static constexpr int W2_STB =
      (W2_HQ + RES::KT2 * T * 128 + RES::KT2 * SF_CHUNK + 4095) / 4096 * 4096;
  static constexpr uint32_t W2_BYTES = RES::KT2 * 16384 + RES::KT2 * T * 128;
  static_assert(W2_ST * W2_STB <= OFF_WSF,
                "the W2 ring lies inside the W13 ring's area");
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
                                       int &chk_slot) {
  using SM_ = Smem<RES>;
  constexpr int W2_ST = SM_::W2_ST, W2_STB = SM_::W2_STB, W2_HQ = SM_::W2_HQ;
  constexpr int OT2 = RES::OT2, KT2 = RES::KT2;
  int const s = k % W2_ST;
  if (k >= W2_ST) {
    mbar_wait(w2_bar<RES>(base, 3 + s), ((k / W2_ST) - 1) & 1);
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
    cnt_wait(g.cnt + RES::CHQ + slot, RES::MT13);
    asm volatile("fence.proxy.async.global;" ::: "memory");
    chk_slot = slot;
  }
  tma3(&maps.m[RES::HQ_KT], f, st + W2_HQ, 0, slot * T, 0, EVICT_LAST);
}

// all 8 warps of the SM, until the queue is empty. gl / gi0 / gi1: ring stage
// counters of the loaders / issuers; wt: accumulator ticks so far (the same on
// entry for every role). Timing build: stage stamps [0] the first entry's first
// stage landed at issuer 0 (the queue's first MMA), [1] issuer 0 takes its
// first W2 entry.
template <class RES>
__device__ __forceinline__ void run_expert_dynamic(G const &g,
                                                   Maps const &maps,
                                                   float *racc,
                                                   uint32_t base,
                                                   uint32_t tb,
                                                   int &gl,
                                                   int &gi0,
                                                   int &gi1,
                                                   int &wt STAGE_STAMP_PARAM) {
  using SM_ = Smem<RES>;
  constexpr int W2_ST = SM_::W2_ST, W2_STB = SM_::W2_STB, W2_HQ = SM_::W2_HQ;
  constexpr int KT_LAT = RES::KT_LAT, MT13 = RES::MT13, OT2 = RES::OT2,
                KT2 = RES::KT2, K = RES::K, LAT = RES::LAT, IR = RES::IR;
  int *const eid_s = RES::eid(), *const ntok_s = RES::ntok(),
             *const tok_s = RES::tok();
  float *const w_s = RES::wt();
  unsigned char *const kk_s = RES::kk();
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    for (int i = 0; i < W2_ST; i++) {
      mbar_init(w2_bar<RES>(base, i), 1);
      mbar_init(w2_bar<RES>(base, 3 + i), 1);
    }
    mbar_init(w2_bar<RES>(base, 6), 1);
    asm volatile("st.shared.u32 [%0], %1;" ::"r"(w2_w7gl<RES>(base)), "r"(gl)
                 : "memory");
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  // W2 tiles n2 = slots x OT2: 2-tile entries, the last t1 = min(the tiles
  // after the first ~4/5 of the slots, NSM) in 1-tile entries, so the SMs
  // finish within about a tile of each other
  int const n_live = s_ints[1], n13 = n_live * MT13, n2 = n_live * OT2;
  int const n4 = ((n2 * 4 / 5) / OT2) * OT2,
            t1 = (n2 - n4 < NSM) ? n2 - n4 : NSM, off1 = n2 - t1,
            n2e = (off1 + 1) / 2;
  int const nentries = n13 + n2e + t1;
  float *acc_s = reinterpret_cast<float *>(rt_sm() + SM_::OFF_ACC);
  auto entry_of =
      [&](int idx, QueueEntry &e) { // entry number idx of the queue order above
        if (idx < n13) {
          e.kind = QE_W13;
          e.a = idx / MT13;
          e.b = idx % MT13;
        } else if (idx < n13 + n2e) {
          int const i0 = 2 * (idx - n13);
          e.kind = QE_W2;
          e.a = i0;
          e.b = (off1 - i0 < 2) ? off1 - i0 : 2;
        } else if (idx < nentries) {
          e.kind = QE_W2;
          e.a = off1 + (idx - n13 - n2e);
          e.b = 1;
        } else {
          e.kind = QE_END;
          e.a = e.b = 0;
        }
      };
  auto w13_job = [&](QueueEntry const &e,
                     TileJob &j) { // the ring job of a W13 entry
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
    j.tile0 = ex * (MT13 * KT_LAT) + e.b * KT_LAT;
  };
  if (warp == 0) {
    if (lane == 0) {
      int qn = 0, w2k = 0,
          chk =
              -1; // w2k: this SM's W2 tiles so far; chk: the slot checked last
      bool switched = false, maps_pf = false;
      int idx = s_ints[2], pend = -1;
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
      publish(e, qn);
      int n13_mine = 0;
      while (e.kind != QE_END) {
        QueueEntry en;
        if (e.kind == QE_W2) {
          int const idx1 =
              pend >= 0 ? pend : (int)atom_inc_lane0(g.cnt + RES::W2NEXT);
          pend = (int)atom_inc_lane0(g.cnt + RES::W2NEXT);
          if (!switched) {
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
          }
          int const slot = e.a / OT2, ex = eid_s[slot], ot0 = e.a - slot * OT2;
          if ((w2k & 1) == 0) {
            scaled_moe::w2_arm<RES>(maps, g, base, w2k, slot, ex, ot0, chk);
          }
          entry_of(idx1, en);
          publish(en, qn + 1);
          for (int i = 1; i < e.b; i++) {
            if (((w2k + i) & 1) == 0) {
              scaled_moe::w2_arm<RES>(
                  maps, g, base, w2k + i, slot, ex, ot0 + i, chk);
            }
          }
          w2k += e.b;
        } else {
          TileJob j;
          w13_job(e, j);
          load_job<128, 16384, false>(j, base, gl, 0, 0, W13_TAKE_AT);
          n13_mine++;
          int const idx1 = (int)atom_inc_lane0(g.cnt + RES::W2NEXT);
          load_job<128, 16384, false>(j, base, gl, 0, W13_TAKE_AT);
          entry_of(idx1, en);
          publish(en, qn + 1);
        }
        e = en;
        qn++;
      }
    }
  } else if (warp == 1 || warp == 6) { // the issuers: the whole warp runs the
                                       // loop, one elected lane issues
    int const iss = warp_same<true>((warp == 1) ? 0 : 1);
    int &gi = (warp == 1) ? gi0 : gi1;
    gi = warp_same<true>(gi);
    uint32_t const tbu = (uint32_t)warp_same<true>((int)tb),
                   baseu = (uint32_t)warp_same<true>((int)base);
    int wtt0 = warp_same<true>(wt), w2k = 0;
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
        for (int i = 0; i < e.b;
             i++, w2k++) { // warp 1: the tile's 4 KT2 MMAs; warp 6: the
                           // second arrival on acc_full
          int const wtt = wtt0 + i, a = wtt & 3, s = w2k % W2_ST;
          if (wtt >= 4) {
            mbar_wait(rt.acc_empty0 + 8 * a, ((wtt >> 2) - 1) & 1);
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
            issue_w2_promoted<IDESC_MX_E4M3, KT2, W2_HQ>(
                baseu + s * W2_STB, tbu + 8 * a, tbu + SM_::CONST_SF);
            tc_commit(w2_bar<RES>(baseu, 3 + s));
            tc_commit(rt.acc_full0 + 8 * a);
          }
        }
        wtt0 += e.b;
      } else { // W13: 8 columns for each piece, PromoRounds ticks
               // (scale_promotion.cuh)
        issue_promoted<IDESC_MX_E4M3, KT_LAT>(
            baseu,
            tbu,
            iss,
            gi,
            wtt0,
            tbu + SM_::CONST_SF STAGE_STAMP_ARG(
                      (iss == 0 && qn == 0) ? stamp : nullptr));
        wtt0 += PromoRounds<KT_LAT>::ROUNDS;
      }
    }
  } else if (warp == 7) { // second loader: the odd W13 stages, the odd W2 tiles
    int w2k7 = 0, chk7 = -1;
    bool ready = false;
    for (int qn = 0;; qn++) {
      int const qpos = qn % W2QD;
      mbar_wait(rt.qfull7_0 + 8 * qpos, (qn / W2QD) & 1);
      QueueEntry const e{w2q7[qpos][0], w2q7[qpos][1], w2q7[qpos][2]};
      __syncwarp();
      if (lane == 0) {
        mbar_arrive(rt.qempty7_0 + 8 * qpos);
      }
      if (e.kind == QE_END) {
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
              scaled_moe::w2_arm<RES>(
                  maps, g, base, w2k7 + i, slot, ex, ot0 + i, chk7);
            }
          }
        }
        w2k7 += e.b;
        __syncwarp();
      } else {
        TileJob j;
        w13_job(e, j);
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
  } else if (warp >= 2 &&
             warp <=
                 5) { // the epilogue, 128 threads (TMEM lane quadrant warp & 3)
    int wtt0 = wt;
    for (int qn = 0;; qn++) {
      int const qpos = qn % W2QD;
      mbar_wait(rt.qfull0 + 8 * qpos, (qn / W2QD) & 1);
      QueueEntry const e{w2q[qpos][0], w2q[qpos][1], w2q[qpos][2]};
      if (e.kind == QE_END) {
        break;
      }
      if (e.kind == QE_W2) { // racc row (t, k) = weight(t) * result, for
                             // the slot's tokens (plain stores)
        for (int item = 0; item < e.b; item++) {
          int const wtt = wtt0 + item, a = wtt & 3, par = (wtt >> 2) & 1,
                    w = e.a + item, slot = w / OT2, ot = w - slot * OT2,
                    nt = ntok_s[slot];
          int const lg = warp & 3, row = lg * 32 + lane;
          float v[8];
          // the weight and h_q scales of the tile, read before the wait
          float sw[2], sh[2][T];
          float const *w2s = buf_at<float const>(g, RES::W2SF) +
                             ((size_t)eid_s[slot] * OT2 + ot) * KT2;
          float const *hs = buf_at<float const>(g, RES::HSF);
          cnt_wait(g.cnt + RES::CHQ + slot,
                   MT13); // acquire: the h_q scales of the slot are ready
#pragma unroll
          for (int kt = 0; kt < 2; kt++) {
            sw[kt] = kt < KT2 ? w2s[kt] : 0.0f;
#pragma unroll
            for (int t = 0; t < T; t++) {
              sh[kt][t] =
                  kt < KT2
                      ? ld_relaxed_f32(hs + ((size_t)slot * T + t) * KT2 + kt)
                      : 0.0f;
            }
          }
          mbar_wait(rt.acc_full0 + 8 * a, par);
          fence_after();
          drain_w2_promoted<KT2>(tb, a, sw, sh, v);
          fence_before();
          mbar_arrive(rt.acc_empty0 + 8 * a);
          for (int i = 0; i < nt; i++) {
            int const t = tok_s[slot * T + i];
            float vt = v[0]; // v[t] by an unrolled select: indexing v with the
                             // run-time t put v in local memory (STL / LDL)
#pragma unroll
            for (int k = 1; k < 8; k++) {
              vt = (t == k) ? v[k] : vt;
            }
            racc[((size_t)t * K + kk_s[slot * T + i]) * LAT + ot * 128 + row] =
                w_s[slot * T + i] * vt; // row (t, k), plain store
          }
        }
        wtt0 += e.b;
      } else { // W13: rows 0..63 gate, 64..127 up of features [64 m, 64 m + 64)
               // -> h_q of the slot
        int const slot = e.a, mt = e.b;
        float v[8];
        // the weight scales, then the rounds
        float *const bs = reinterpret_cast<float *>(rt_sm() + SM_::OFF_BS);
        stage_w13_scales<KT_LAT, IR>(
            buf_at<float const>(g, RES::W13SF), eid_s[slot], mt, bs);
        int const row = drain_promoted<KT_LAT, GateUpScales>(
            tb,
            wtt0,
            bs,
            reinterpret_cast<float const *>(rt_sm() + SM_::OFF_XS),
            v);
        for (int t = 0; t < T; t++) {
          acc_s[row * 8 + t] = v[t];
        }
        asm volatile("bar.sync 1, 128;" ::: "memory");
        // h (fp32) = SiLU(gate) up -> h_q, one fp32 scale for each (token,
        // 128 features)
        float h[T];
        if (row < 64) {
#pragma unroll
          for (int t = 0; t < T; t++) {
            h[t] = silu2(acc_s[row * 8 + t], acc_s[(64 + row) * 8 + t]);
          }
        }
        float *const h32 = buf_at<float>(g, RES::H32);
        h_quant_pair<IR>(h32,
                         h32 + (size_t)RES::NSLOT * T * IR,
                         buf_at<uint8_t>(g, RES::HQ),
                         buf_at<float>(g, RES::HSF),
                         g.cnt + RES::PAIRC,
                         g.cnt + RES::CHQ,
                         slot,
                         mt,
                         row,
                         h);
        wtt0 += PromoRounds<KT_LAT>::ROUNDS;
      }
      mbar_arrive(rt.qempty0 + 8 * qpos); // 128 arrivals: entry retired
    }
  }
}

// ---- the switch into the expert phase ----
// fresh ring barriers and stage counters; the constant scale chunk (UE8M0 1,
// for both MMA operands) -> TMEM column CONST_SF (warp 1); the NPAIR routing
// pairs (topk_route's buffer) -> shared memory (warps 2..5, table_poll)
template <class RES>
__device__ __forceinline__ void
    expert_phase_switch(unsigned long long const *pairs,
                        uint32_t base,
                        uint32_t tb,
                        int &gl,
                        int &gi0,
                        int &gi1) {
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    ring_reinit(false);
  }
  gl = gi0 = gi1 = 0;
  if (threadIdx.x < SF_CHUNK / 4) { // the constant scale chunk (UE8M0 1)
    const_sf_fill(base + Smem<RES>::OFF_CSF, threadIdx.x);
  }
  __syncthreads();
  if (warp == 1 && lane == 0) {
    cp_sf(tb + Smem<RES>::CONST_SF, base + Smem<RES>::OFF_CSF);
    tc_commit(rt.misc0 + 16);
  }
  if (warp >= 2 &&
      warp <=
          5) { // the 128 epilogue threads: one pair each (more pairs: in turn)
    if constexpr (RES::NPAIR == 128) {
      table_poll<RES>(pairs, threadIdx.x - 64);
    } else {
      for (int i = threadIdx.x - 64; i < RES::NPAIR; i += 128) {
        table_poll<RES>(pairs, i);
      }
    }
  }
  mbar_wait(rt.misc0 + 16, 0);
  fence_after();
  __syncthreads();
}

// wait until z_q (all K tiles) and its fp32 scales have landed (group_quant,
// this GPU's copy): each thread polls its 16-B pieces (0xFF-prefilled)
template <class RES>
__device__ __forceinline__ void wait_z_landed(G const &g) {
  constexpr int NZV = (int)(zq32_bytes(RES::LAT) / 16), PER = (NZV + 255) / 256;
  bool ok[PER];
  bool all;
#pragma unroll
  for (int q = 0; q < PER; q++) {
    ok[q] = (threadIdx.x + 256 * q) >= NZV;
  }
  do {
    uint4 u[PER];
#pragma unroll
    for (int q = 0; q < PER; q++) {
      if (!ok[q]) {
        u[q] = ld16_relaxed(g.rv + RES::ZOFF +
                            (size_t)(threadIdx.x + 256 * q) * 16 + rg_set());
      }
    }
    all = true;
#pragma unroll
    for (int q = 0; q < PER; q++) {
      if (!ok[q]) {
        ok[q] = valid16(u[q]);
      }
      all = all && ok[q];
    }
  } while (!all);
}

} // namespace scaled_moe

// the node's scaled_moe::Slots: its buffer slots (scaled_moe_experts_layer:
// h_q, its fp32 scales, the w13 scales (a tensor), the scratch of h, the w2
// scales (a tensor)), its map slots (w13, w13 x2, w2, w2 x2, z_q, z_q x2 K
// tiles (each 2 slots: the exchange region's sets), h_q), its counter, z_q's
// offset, the sizes: NE and K from topk_route's params (PAIRS), LAT from
// group_quant's (ZQ), IR its own params[0]
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class ZQ,
          class PAIRS>
using ScaledMoeSlotsOf = scaled_moe::Slots<slot_at<BUF_SLOTS, 0>(),
                                           slot_at<BUF_SLOTS, 1>(),
                                           SELF::counter,
                                           slot_at<MAP_SLOTS, 0>(),
                                           slot_at<MAP_SLOTS, 1>(),
                                           slot_at<BUF_SLOTS, 2>(),
                                           slot_at<BUF_SLOTS, 3>(),
                                           slot_at<MAP_SLOTS, 2>(),
                                           slot_at<MAP_SLOTS, 3>(),
                                           slot_at<BUF_SLOTS, 4>(),
                                           slot_at<MAP_SLOTS, 4>(),
                                           slot_at<MAP_SLOTS, 5>(),
                                           slot_at<MAP_SLOTS, 6>(),
                                           exchange_offset[ZQ::buf],
                                           PAIRS::v[0],
                                           PAIRS::v[1],
                                           ZQ::v[0],
                                           PARAMS::v[0]>;

// scaled_moe_experts task (one per SM, task x on SM x): wait for z_q, phase
// switch, routing table, then the queue (above) -> the routed rows [T][K][LAT]
// in the node's output buffer; its counter counts the SMs done. IN: its
// inputs' producers, in the layer's order (z_q from group_quant, pairs from
// topk_route, ...). PARAMS {IR}
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
      ZQ::buf >= 0 && exchange_bytes[ZQ::buf] == zq32_bytes(RES::LAT),
      "scaled_moe_experts: z_q in the exchange region (group_quant's)");
  int const sm_id = L.sm_id;
  __syncthreads(); // every warp is done with the placed tiles (the loader runs
                   // ahead of the landings)
  uint32_t pre_idx = 0;
  if (threadIdx.x == 0) {
    pre_idx = atom_inc_lane0(g.cnt + RES::W2NEXT);
  }
  for (int i = threadIdx.x; i < RES::NE; i += 256) {
    RES::cnt_e()[i] = 0;
  }
  scaled_moe::wait_z_landed<RES>(g);
  // copy the fp32 scales of z_q to shared memory. Each thread polled only its
  // own 16-B pieces: wait for all threads first.
  __syncthreads();
  // z_q came from generic stores (group_quant); the TMA loads read it next
  asm volatile("fence.proxy.async.global;" ::: "memory");
  scaled_moe::stage_x_scales<RES::KT_LAT>(
      g.rv + RES::ZOFF + (size_t)T * RES::LAT + rg_set(),
      reinterpret_cast<float *>(rt_sm() + scaled_moe::Smem<RES>::OFF_XS));
  if (threadIdx.x == 0) {
    g.stamps[sm_id * NSTAMP + STAMP_TASK0] = gtime();
  }
  if (!L.ring_relaid) {
    scaled_moe::expert_phase_switch<RES>(
        buf_at<unsigned long long const>(g, PAIRS::buf),
        L.base,
        L.tb,
        L.gl,
        L.gi0,
        L.gi1);
    L.ring_relaid = true;
  }
  build_table<RES>();
  if (threadIdx.x == 0) {
    s_ints[2] = (int)pre_idx;
  }
  __syncthreads();
  scaled_moe::run_expert_dynamic<RES>(g,
                                      maps,
                                      buf_at<float>(g, SELF::buf),
                                      L.base,
                                      L.tb,
                                      L.gl,
                                      L.gi0,
                                      L.gi1,
                                      L.wt STAGE_STAMP_ARG(L.stage_stamps));
  if (threadIdx.x == 0) {
    g.stamps[sm_id * NSTAMP + STAMP_TASK1] = gtime();
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(g.cnt + SELF::counter, 1); // this SM's routed rows are written (its
                                       // consumer waits for all SMs)
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
