// tasks/moe_experts.cuh -- the routed experts, after routing. Every SM takes
// entries from one queue (the counter W2NEXT: an atomic add returns the next
// entry index), in this order:
//   1. W13 items: (expert slot, 64-feature tile m): gate and up rows of the
//   expert x z_q (MXFP4 x e4m3, K = LAT), then
//      h_q = MXFP8(SiTU(gate, up)) for the tokens of that expert, then counter
//      CHQ[slot] += 1 (release)
//   2. W2 entries of 2 tiles, the last NSM tiles in 1-tile entries: a tile =
//   (slot, 128-row output tile) = W2 x h_q (K = IR), then
//      for each of the expert's tokens t (the expert is the k-th of t's K):
//      racc row (t, k) = weight(t, expert) * result, one writer per value
//      (allreduce_send adds a token's K rows in the order k = 0 .. K - 1: the
//      same sum every launch)
// Roles: warp 0 lane 0 takes the entries (two ahead), publishes each to w2q /
// w2q7 and arms its ring stages; warp 7 arms the other stages; warps 1 and 6
// issue the MMAs (whole warps, one elected lane issues); warps 2..5 drain the
// accumulators and write results. W13 runs through the ring of runtime.cuh
// (SMAX stages of two pieces; warp 0 the even stages, warp 7 the odd ones, each
// issuer one piece of every stage). At its first W2 entry an SM lays a 3-stage
// ring of one W2 tile per stage over the W13 ring's area: warps 0 / 7 load the
// even / odd tiles, warp 1 issues every tile, warp 6 gives acc_full its second
// arrival, the epilogue drains one accumulator half (w2_arm and below). TMEM
// columns: 0..63 accumulators (runtime.cuh issue_job), 72 + 4 kt the KT_LAT z_q
// scale chunks (phase switch), 232.. the W13 weight scales per (ring stage,
// piece), from 272 / 320 the W2 weight / h_q scales per (W2 stage, K tile)
// (ExpertSmem). Also the routing table of the step (build_table): slots = the
// experts used, in ascending id, each with its tokens and weights. Sizes
// (ExpertSlots): NE experts and K of them per token (topk_route's params), LAT
// = z_q's width (sum_quant_send's params), IR = the expert intermediate on this
// GPU (the node's params); the tile counts below follow from them.
#pragma once
#include "../core.cuh"
#include "sum_quant_send.cuh" // the layout of its input z_q

namespace static_mk {

// one entry of moe_experts' work queue (run_expert_dynamic); stored as 3 ints
// in the shared-memory copies w2q / w2q7
enum QueueKind { QE_END = -1, QE_W13 = 1, QE_W2 = 2 };
struct QueueEntry {
  int kind; // QueueKind
  int a, b; // QE_W13: expert slot, 64-feature tile m (0..MT13-1)
            // QE_W2: first tile (slot * OT2 + output tile), number of tiles (1
            // or 2, never crossing a slot)
};

// what the bodies below use of the layer state, as slots and offsets
// (compile-time; run_moe_experts fills it from the node's slots): buffers HQ,
// HSF (h_q and its scales, the node's scratch); counters CNT (the node's first:
// SMs done), W2NEXT = CNT + 32 (the queue's next entry), CHQ = CNT + 64 .. (per
// expert slot the W13 items that wrote their h_q); tensor maps (Maps::m) of the
// expert weights and scales, of z_q (ZQ, ZQ2, XSF28: set 0, set 1 at + 1) and
// of h_q (HQ3, HSF3); ZOFF: z_q's offset in the exchange region; the sizes NE,
// K, LAT, IR and the tile counts from them
template <int HQ_,
          int HSF_,
          int CNT_,
          int W13_,
          int W13X2_,
          int W13SF_,
          int W13SFX2_,
          int W2_,
          int W2X2_,
          int W2SF_,
          int W2SFX2_,
          int ZQ_,
          int ZQ2_,
          int XSF28_,
          int HQ3_,
          int HSF3_,
          size_t ZOFF_,
          int NE_,
          int K_,
          int LAT_,
          int IR_>
struct ExpertSlots {
  static constexpr int HQ = HQ_, HSF = HSF_, CNT = CNT_, W2NEXT = CNT_ + 32,
                       CHQ = CNT_ + 64;
  static constexpr int W13 = W13_, W13X2 = W13X2_, W13SF = W13SF_,
                       W13SFX2 = W13SFX2_, W2 = W2_, W2X2 = W2X2_, W2SF = W2SF_,
                       W2SFX2 = W2SFX2_;
  static constexpr int ZQ = ZQ_, ZQ2 = ZQ2_, XSF28 = XSF28_, HQ3 = HQ3_,
                       HSF3 = HSF3_;
  static constexpr size_t ZOFF = ZOFF_;
  static constexpr int NE = NE_, K = K_, LAT = LAT_, IR = IR_;
  static constexpr int NPAIR = T * K; // routing pairs of a step
  static constexpr int NSLOT =
      T * K; // expert slots: at most every pair its own expert
  static constexpr int KT_LAT = LAT / 128; // K tiles of z_q (W13's K)
  static constexpr int MT13 =
      IR /
      64; // W13 items per expert: 64 features each (64 gate rows + 64 up rows)
  static constexpr int OT2 = LAT / 128; // W2 output tiles per expert
  static constexpr int KT2 = IR / 128;  // W2 K tiles
  static_assert(LAT % 128 == 0 && IR % 128 == 0,
                "moe_experts: whole 128-wide tiles");
  static_assert(
      NE % 32 == 0 && NE <= 1024 && NPAIR <= 256,
      "moe_experts: build_table's 8 warps x 128 experts, a thread per pair");
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

constexpr int W13_TAKE_AT =
    8; // a W13 item takes the SM's next queue entry once this many of its
       // stages are armed (late, so that the first items spread over the SMs:
       // run_expert_dynamic)

// one routing pair (token i / K, k = i % K) per call, i < NPAIR: poll it
// (0xFF-prefilled), record it, count the expert's tokens
template <class RES>
__device__ __forceinline__ void table_poll(unsigned long long const *pairs,
                                           int i) {
  int *const cntE_s = RES::cnt_e(), *const sel_ss = RES::sel();
  float *const wsel_ss = RES::wsel();
  unsigned long long v;
  do {
    asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];"
                 : "=l"(v)
                 : "l"(pairs + i)
                 : "memory");
  } while (v == ~0ull);
  int const e = (int)(v >> 32);
  sel_ss[i] = e;
  wsel_ss[i] = __uint_as_float((uint32_t)v);
  atomicAdd(&cntE_s[e], 1);
}

// after table_poll of all NPAIR pairs, 256 threads: slots = the experts with
// tokens, in ascending id (parallel prefix count); per slot its tokens and
// weights; s_ints[1] = number of slots
template <class RES>
__device__ __forceinline__ void build_table() {
  constexpr int NE = RES::NE, K = RES::K, NPAIR = RES::NPAIR;
  int *const cntE_s = RES::cnt_e(), *const sel_ss = RES::sel(),
             *const eid_s = RES::eid(), *const ntok_s = RES::ntok(),
             *const tok_s = RES::tok();
  float *const wsel_ss = RES::wsel(), *const w_s = RES::wt();
  unsigned char *const kk_s = RES::kk();
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __syncthreads();
  // warp w looks at experts [128 w, 128 w + 128): four 32-expert blocks b, one
  // bit per lane
  int cnt_q[4];
  uint32_t m_q[4];
#pragma unroll
  for (int q = 0; q < 4; q++) {
    int const b = warp * 4 + q;
    bool const live = (b < NE / 32) && cntE_s[b * 32 + lane] > 0;
    m_q[q] = __ballot_sync(0xffffffffu, live);
    cnt_q[q] = __popc(m_q[q]);
  }
  if (lane == 0) {
    s_ints[16 + warp] = cnt_q[0] + cnt_q[1] + cnt_q[2] + cnt_q[3];
  }
  __syncthreads();
  int wbase = 0;
  for (int w_ = 0; w_ < warp; w_++) {
    wbase += s_ints[16 + w_];
  }
  int n_live = 0;
  for (int w_ = 0; w_ < 8; w_++) {
    n_live += s_ints[16 + w_];
  }
  {
    int run = wbase;
#pragma unroll
    for (int q = 0; q < 4; q++) {
      int const b = warp * 4 + q;
      if (b < NE / 32) {
        if (cntE_s[b * 32 + lane] > 0) {
          int const slot = run + __popc(m_q[q] & ((1u << lane) - 1u));
          eid_s[slot] = b * 32 + lane;
          ntok_s[slot] = 0;
        }
        run += cnt_q[q];
      }
    }
  }
  if (threadIdx.x == 0) {
    s_ints[1] = n_live;
  }
  __syncthreads();
  if (threadIdx.x < NPAIR) { // one pair per thread: its slot by binary search
                             // over eid_s, its position by a shared atomic
    int const t = threadIdx.x / K, e = sel_ss[threadIdx.x];
    int lo = 0, hi = n_live - 1;
    while (lo < hi) {
      int const mid = (lo + hi) >> 1;
      if (eid_s[mid] < e) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    int const pos_ = atomicAdd(&ntok_s[lo], 1);
    tok_s[lo * T + pos_] = t;
    w_s[lo * T + pos_] = wsel_ss[threadIdx.x];
    kk_s[lo * T + pos_] =
        (unsigned char)(threadIdx.x %
                        K); // k: the pair's place in the token's K (its R row)
  }
  __syncthreads();
}

// ---- shared memory after the ring (offsets from the dynamic base; keep this
// layout: the layer's time depends on where these
//      are): OFF_HQ the W2 ring's barriers (w2_bar), then KT2 x 1 KB and KT2
//      scale chunks not used, OFF_ACC a W13 item's accumulator (128 rows x T
//      fp32, for SiTU across rows). The z scale chunks are staged once at the
//      phase switch in the (then idle) ring area (OFF_XSF) ----
template <class RES>
struct ExpertSmem {
  static constexpr int OFF_XSF = OFF_W, OFF_HQ = RING_BYTES,
                       OFF_HSF = OFF_HQ + RES::KT2 * 1024,
                       OFF_ACC = OFF_HSF + RES::KT2 * SF_CHUNK;
  static constexpr int BYTES =
      OFF_ACC + 128 * T * 4; // the node's need from the base
  static_assert(OFF_HQ % 1024 == 0, "128-B swizzled tiles are 1024-aligned");
  static_assert(RES::KT_LAT * SF_CHUNK <= OFF_WSF,
                "the z scale chunks in the ring area");
  // ---- W2 ring: one tile per stage (KT2 x 16 KB weight pieces | KT2 scale
  // chunks | h_q KT2 KB | h_q scales KT2 chunks) ----
  static constexpr int W2_ST = 3;
  static constexpr int W2_WSF = RES::KT2 * 16384,
                       W2_HQ =
                           (W2_WSF + RES::KT2 * SF_CHUNK + 1023) / 1024 * 1024;
  static constexpr int W2_HSF = W2_HQ + RES::KT2 * T * 128,
                       W2_STB =
                           (W2_HSF + RES::KT2 * SF_CHUNK + 4095) / 4096 * 4096;
  static constexpr uint32_t W2_BYTES = RES::KT2 * 8192 + RES::KT2 * SF_CHUNK +
                                       RES::KT2 * T * 128 + RES::KT2 * SF_CHUNK;
  static_assert(W2_ST * W2_STB <= OFF_WSF,
                "the W2 ring lies inside the W13 ring's area");
  // TMEM columns: z_q scales 72 + 4 kt, W13 weight scales 232.. (runtime.cuh
  // issue_job_warp SF_SLOTS), W2 weight / h_q scales SFA_W2 / SFB_W2 + 4 KT2 s
  // + 4 kt
  static constexpr int SFA_W2 = 272, SFB_W2 = 320;
  static_assert(72 + 4 * RES::KT_LAT <= 232 &&
                    SFA_W2 + 4 * RES::KT2 * W2_ST <= SFB_W2 &&
                    SFB_W2 + 4 * RES::KT2 * W2_ST <= 512,
                "moe_experts: the TMEM columns of the scales");
};
// mbarriers and one word in the (otherwise unused) OFF_HQ area: full [3], empty
// [3], ready (warp 0 has re-laid the ring), then warp 7's ring stage counter
// (W13 stages it has armed)
template <class RES>
__device__ __forceinline__ uint32_t w2_bar(uint32_t base, int i) {
  return base + ExpertSmem<RES>::OFF_HQ + 8 * i;
}
template <class RES>
__device__ __forceinline__ uint32_t w2_w7gl(uint32_t base) {
  return base + ExpertSmem<RES>::OFF_HQ + 64;
}
// arm W2 tile k of this SM: wait for its ring slot, load the weights and
// scales, and once all W13 items of the tile's expert are done (CHQ counter,
// acquire, then the async-proxy fence) its h_q and scales. chk_slot: the slot
// this loader has checked last
template <class RES>
__device__ __forceinline__ void w2_arm(Maps const &maps,
                                       G const &g,
                                       uint32_t base,
                                       int k,
                                       int slot,
                                       int ex,
                                       int ot,
                                       int &chk_slot) {
  using SM_ = ExpertSmem<RES>;
  constexpr int W2_ST = SM_::W2_ST, W2_STB = SM_::W2_STB, W2_WSF = SM_::W2_WSF,
                W2_HQ = SM_::W2_HQ, W2_HSF = SM_::W2_HSF;
  constexpr int OT2 = RES::OT2, KT2 = RES::KT2;
  int const s = k % W2_ST;
  if (k >= W2_ST) {
    mbar_wait(w2_bar<RES>(base, 3 + s), ((k / W2_ST) - 1) & 1);
  }
  uint32_t const st = base + s * W2_STB, f = w2_bar<RES>(base, s);
  int const p0 = ex * (OT2 * KT2) + ot * KT2;
  mbar_expect(f, SM_::W2_BYTES);
  // the KT2 weight pieces and their scale chunks: two per box, the last one
  // alone when KT2 is odd
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
#pragma unroll
  for (int kp = 0; kp + 1 < KT2; kp += 2) {
    tma3(&maps.m[RES::W2SFX2],
         f,
         st + W2_WSF + kp * SF_CHUNK,
         0,
         0,
         p0 + kp,
         EVICT_FIRST);
  }
  if constexpr (KT2 % 2 == 1) {
    tma3(&maps.m[RES::W2SF],
         f,
         st + W2_WSF + (KT2 - 1) * SF_CHUNK,
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
  tma3(&maps.m[RES::HQ3], f, st + W2_HQ, 0, slot * T, 0, EVICT_LAST);
  tma3(&maps.m[RES::HSF3], f, st + W2_HSF, 0, 0, slot * KT2, EVICT_LAST);
}

// all 8 warps of the SM, until the queue is empty. gl / gi0 / gi1: ring stage
// counters of the loaders / issuers; wt: accumulator tiles so far (the same on
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
  using SM_ = ExpertSmem<RES>;
  constexpr int W2_ST = SM_::W2_ST, W2_STB = SM_::W2_STB, W2_WSF = SM_::W2_WSF,
                W2_HQ = SM_::W2_HQ, W2_HSF = SM_::W2_HSF;
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
    j.sfmap = &maps.m[RES::W13SF];
    j.wmap2 = &maps.m[RES::W13X2];
    j.sfmap2 = &maps.m[RES::W13SFX2];
    j.amap = &maps.m[RES::ZQ + dbuf_par];
    j.amap2 = &maps.m[RES::ZQ2 + dbuf_par];
    j.tile0 = j.chunk0 = ex * (MT13 * KT_LAT) + e.b * KT_LAT;
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
          prefetch_tmap(&maps.m[RES::W2SFX2]);
          prefetch_tmap(&maps.m[RES::W2SF]);
          prefetch_tmap(&maps.m[RES::HQ3]);
          prefetch_tmap(&maps.m[RES::HSF3]);
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
            // the W13 ring's stages are all consumed before the W2 ring
            // overwrites their area. Stage st0's last use u (global stage gq)
            // completes `empty` phase u. Warp 7 may not have armed its last
            // stages yet, and then use u - 1 of such a stage may still be in
            // flight: the barrier one phase behind, the other parity, and a
            // parity wait for u passes at once (wrong W13 results). So first
            // every stage < gl armed (warp 7's counter): arming u needed u - 1
            // done, the barrier is at phase u or u + 1, and the parity wait for
            // u is exact.
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
            w2_arm<RES>(maps, g, base, w2k, slot, ex, ot0, chk);
          }
          entry_of(idx1, en);
          publish(en, qn + 1);
          for (int i = 1; i < e.b; i++) {
            if (((w2k + i) & 1) == 0) {
              w2_arm<RES>(maps, g, base, w2k + i, slot, ex, ot0 + i, chk);
            }
          }
          w2k += e.b;
        } else {
          TileJob j;
          w13_job(e, j);
          load_job(j, base, gl, 0, 0, W13_TAKE_AT);
          n13_mine++;
          int const idx1 = (int)atom_inc_lane0(g.cnt + RES::W2NEXT);
          load_job(j, base, gl, 0, W13_TAKE_AT);
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
             i++, w2k++) { // warp 1: the tile's 12 MMAs; warp 6: acc_full's
                           // second arrival
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
          if (issuer_lane<true>()) {
            uint32_t const st = baseu + s * W2_STB, acc = tbu + 8 * a;
#pragma unroll
            for (int kt = 0; kt < KT2; kt++) {
              uint32_t const sfa = tbu + SM_::SFA_W2 + 4 * KT2 * s + 4 * kt,
                             sfb = tbu + SM_::SFB_W2 + 4 * KT2 * s + 4 * kt;
              cp_sf(sfa, st + W2_WSF + kt * SF_CHUNK);
              cp_sf(sfb, st + W2_HSF + kt * SF_CHUNK);
              uint64_t const dw = mkdesc(st + kt * 16384),
                             dh = mkdesc(st + W2_HQ) + 64 * kt;
              mma_mx(acc, dw, dh, idesc_mx(0), kt == 0 ? 0u : 1u, sfa, sfb);
              mma_mx(acc, dw + 2, dh + 2, idesc_mx(1), 1u, sfa, sfb);
              mma_mx(acc, dw + 4, dh + 4, idesc_mx(2), 1u, sfa, sfb);
              mma_mx(acc, dw + 6, dh + 6, idesc_mx(3), 1u, sfa, sfb);
            }
            tc_commit(w2_bar<RES>(baseu, 3 + s));
            tc_commit(rt.acc_full0 + 8 * a);
          }
        }
        wtt0 += e.b;
      } else { // W13
        TileJob j;
        w13_job(e, j);
        int const a = wtt0 & 3;
        if (wtt0 >= 4) {
          mbar_wait(rt.acc_empty0 + 8 * a, ((wtt0 >> 2) - 1) & 1);
        }
        issue_job_warp<128, true, true>(
            j,
            baseu,
            tbu,
            iss,
            gi,
            a,
            tbu + 72 STAGE_STAMP_ARG((iss == 0 && qn == 0) ? stamp : nullptr));
        wtt0++;
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
              w2_arm<RES>(maps, g, base, w2k7 + i, slot, ex, ot0 + i, chk7);
            }
          }
        }
        w2k7 += e.b;
        __syncwarp();
      } else {
        TileJob j;
        w13_job(e, j);
        if (lane == 0) {
          load_job(j, base, gl, 1);
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
      if (e.kind == QE_W2) { // R[t][128 ot + row] += weight(t) * result, for
                             // the slot's tokens
        for (int item = 0; item < e.b; item++) {
          int const wtt = wtt0 + item, a = wtt & 3, par = (wtt >> 2) & 1,
                    w = e.a + item, slot = w / OT2, ot = w - slot * OT2,
                    nt = ntok_s[slot];
          int const lg = warp & 3, row = lg * 32 + lane;
          float v[8];
          mbar_wait(rt.acc_full0 + 8 * a, par);
          fence_after();
          tmem_ld8(tb + 8 * a + ((uint32_t)(lg * 32) << 16),
                   v); // issuer 0's half: it wrote the whole tile
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
        int const slot = e.a, mt = e.b, a = wtt0 & 3, par = (wtt0 >> 2) & 1;
        float v[8];
        int const row = drain_acc(tb, a, par, v);
        for (int t = 0; t < T; t++) {
          acc_s[row * 8 + t] = v[t];
        }
        asm volatile("bar.sync 1, 128;" ::: "memory");
        if (row < 64) {
          int const f = mt * 64 + row, kt2 = f >> 7, kb = (f >> 5) & 3;
          float h[T], am[T];
#pragma unroll
          for (int t = 0; t < T; t++) {
            h[t] = situ2(acc_s[row * 8 + t], acc_s[(64 + row) * 8 + t]);
            am[t] = fabsf(h[t]);
          }
#pragma unroll
          for (int d = 16; d > 0; d >>= 1)
#pragma unroll
            for (int t = 0; t < T; t++) {
              am[t] = fmaxf(am[t], __shfl_xor_sync(0xffffffffu, am[t], d));
            }
          uint8_t q[T], sc[T];
#pragma unroll
          for (int t = 0; t < T; t++) {
            int ex = ceil_log2_bits(fmaxf(am[t], 1.0e-30f) * (1.0f / 448.0f));
            ex = ex < -127 ? -127 : (ex > 127 ? 127 : ex);
            q[t] =
                cvt_e4m3_nv(h[t] * __uint_as_float((uint32_t)(127 - ex) << 23));
            sc[t] = (uint8_t)(ex + 127);
          }
#pragma unroll
          for (int t = 0; t < T; t++) {
            buf_at<uint8_t>(g, RES::HQ)[((size_t)slot * T + t) * IR + f] = q[t];
            if (lane == 0) {
              buf_at<uint8_t>(g,
                              RES::HSF)[((size_t)slot * KT2 + kt2) * SF_CHUNK +
                                        t * 16 + kb] = sc[t];
            }
          }
        }
        // no per-thread fence: the bar.sync orders every thread's h_q / scale
        // stores before thread 64's red.release.gpu (cumulative; PTX ISA memory
        // model: bar.sync synchronizes, release patterns, causality order). A
        // __threadfence (fence.sc.gpu) here would stall the SM's next global
        // access for microseconds.
        asm volatile("bar.sync 1, 128;" ::: "memory");
        if (threadIdx.x == 64) {
          cnt_add(g.cnt + RES::CHQ + slot,
                  1); // release; the W2 loaders acquire MT13 of these per slot
        }
        wtt0++;
      }
      mbar_arrive(rt.qempty0 + 8 * qpos); // 128 arrivals: entry retired
    }
  }
}

// ---- the switch into the expert phase ----
// fresh ring barriers and stage counters; the KT_LAT z_q scale chunks -> TMEM
// columns 72.. (warp 1); the NPAIR routing pairs (topk_route's buffer) ->
// shared memory (warps 2..5, table_poll)
template <class RES>
__device__ __forceinline__ void
    expert_phase_switch(unsigned long long const *pairs,
                        Maps const &maps,
                        uint32_t base,
                        uint32_t tb,
                        int &gl,
                        int &gi0,
                        int &gi1) {
  constexpr int KT_LAT = RES::KT_LAT, OFF_XSF = ExpertSmem<RES>::OFF_XSF;
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    ring_reinit(false);
  }
  gl = gi0 = gi1 = 0;
  __syncthreads();
  if (threadIdx.x == 0) {
    mbar_expect(rt.misc0, KT_LAT * SF_CHUNK);
    tma3(&maps.m[RES::XSF28 + dbuf_par],
         rt.misc0,
         base + OFF_XSF,
         0,
         0,
         0,
         EVICT_LAST);
  }
  mbar_wait(rt.misc0, 0);
  fence_after();
  if (warp == 1 && lane == 0) {
    for (int kt = 0; kt < KT_LAT; kt++) {
      cp_sf(tb + 72 + 4 * kt, base + OFF_XSF + kt * SF_CHUNK);
    }
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

// wait until z_q (all K tiles) and its scale chunks from every GPU have landed:
// each thread polls its 16-B pieces (0xFF-prefilled)
template <class RES>
__device__ __forceinline__ void wait_z_landed(G const &g) {
  constexpr int NZV = (int)(zq_bytes(RES::LAT) / 16), PER = (NZV + 255) / 256;
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

// the node's ExpertSlots: its slots (moe_experts_layer: h_q, its scales; the
// maps w13, w13 x2, w13 scales, x2, w2, w2 x2, w2 scales, x2, z_q, z_q x2 K
// tiles, z_q's scale chunks (each 2 slots: the exchange region's sets), h_q,
// h_q's scales), its counter, z_q's offset, the sizes: NE and K from
// topk_route's params (PAIRS), LAT from sum_quant_send's (ZQ), IR its own
// params[0]
template <class SELF, class PARAMS, class SLOTS, class ZQ, class PAIRS>
using ExpertSlotsOf = ExpertSlots<slot_at<SLOTS, 0>(),
                                  slot_at<SLOTS, 1>(),
                                  SELF::counter,
                                  slot_at<SLOTS, 2>(),
                                  slot_at<SLOTS, 3>(),
                                  slot_at<SLOTS, 4>(),
                                  slot_at<SLOTS, 5>(),
                                  slot_at<SLOTS, 6>(),
                                  slot_at<SLOTS, 7>(),
                                  slot_at<SLOTS, 8>(),
                                  slot_at<SLOTS, 9>(),
                                  slot_at<SLOTS, 10>(),
                                  slot_at<SLOTS, 11>(),
                                  slot_at<SLOTS, 12>(),
                                  slot_at<SLOTS, 13>(),
                                  slot_at<SLOTS, 14>(),
                                  exchange_offset[ZQ::buf],
                                  PAIRS::v[0],
                                  PAIRS::v[1],
                                  ZQ::v[0],
                                  PARAMS::v[0]>;

// moe_experts task (one per SM, task x on SM x): wait for z_q, phase switch,
// routing table, then the queue (above) -> the routed rows [T][K][LAT] in the
// node's output buffer; its counter counts the SMs done. IN: its inputs'
// producers, in the layer's order (z_q, pairs, h_s, ...). PARAMS {IR}
template <class SELF,
          class PARAMS,
          class SLOTS,
          class ZQ,
          class PAIRS,
          class HS,
          class... REST>
__device__ __forceinline__ void run_moe_experts(Maps const &maps,
                                                G const &g,
                                                KernelLocals &L,
                                                StaticTask const &) {
  static_assert(
      SELF::buf >= 0 && SELF::counter >= 0 && PAIRS::buf >= 0,
      "moe_experts: its output rows, its counter, topk_route's pairs");
  static_assert(ZQ::buf >= 0 && exchange_bytes[ZQ::buf] == zq_bytes(ZQ::v[0]),
                "moe_experts: z_q in the exchange region");
  using RES = ExpertSlotsOf<SELF, PARAMS, SLOTS, ZQ, PAIRS>;
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
  wait_z_landed<RES>(g);
  if (threadIdx.x == 0) {
    g.stamps[sm_id * NSTAMP + STAMP_TASK0] = gtime();
  }
  if (!L.ring_relaid) {
    expert_phase_switch<RES>(buf_at<unsigned long long const>(g, PAIRS::buf),
                             maps,
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
  run_expert_dynamic<RES>(g,
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

// dynamic shared memory: the ring, then ExpertSmem's
template <class SELF,
          class PARAMS,
          class SLOTS,
          class ZQ,
          class PAIRS,
          class... REST>
constexpr int smem_moe_experts() {
  return ExpertSmem<ExpertSlotsOf<SELF, PARAMS, SLOTS, ZQ, PAIRS>>::BYTES;
}

} // namespace static_mk
