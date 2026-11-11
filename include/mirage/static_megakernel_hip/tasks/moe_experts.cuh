// tasks/moe_experts.cuh (HIP, gfx950) -- the routed experts, after routing; the
// same task as include/mirage/static_megakernel/tasks/moe_experts.cuh (one task
// per CU, task x on CU x; its output: the routed rows [T][K][LAT] fp32, one
// writer per value; its counter counts the CUs done), a new body for gfx950
// (amd/notes/moe_experts_gfx950.md; the kernels of amd/bench/expert_unit.hip):
//   1. z_q (sum_quant_send's output, this GPU's exchange region: polled, system
//      scope) -> LDS in the scaled MFMA's fragment order: [K steps][64 lanes]
//      [32 B] (lane = token + 16 g) and its scales [K steps][64 lanes]
//   2. the routing table (every CU builds the same one): poll the T x K pairs
//      (topk_route's output), slots = the experts used in ascending id, per slot
//      its tokens (position p = its column in W2's B operand), weights, k
//   3. the W13 items (slot, 32 features: 4 row tiles x LAT / 128 K steps of
//      v_mfma_scale_f32_16x16x128_f8f6f4, FP4 weights A, FP8 tokens B) from 8
//      queues per GPU, one per die (agent-scope counters in buffer slot 2; each
//      wavefront takes one at a time, from its die's queue, then from any that
//      is not empty): SiTU, MXFP8 h_q of the slot's tokens -> HQ / HSF (st8_out; HQ
//      compact: only the T token columns of each lane group, [slot][IR / 128]
//      [4 x T][32 B] = the NSLOT x T x IR bytes Python declares; HQ and HSF
//      0xFF before each launch). Then the W2 items (slot, W2_TILES row tiles of
//      16 x IR / 128 K steps), from 8 more queues, one per die, the same way
//      (the next one's ticket taken before the current one's loads); each polls
//      the slot's h_q itself until no byte it uses is 0xFF, -> rows (t, k) of
//      the output = weight x result
//      (st16_out). The waits are safe: a W2 item waits only for W13 items, which
//      are taken only by waves already in this task, which finish them without
//      waiting for anything. (Dealing the W13 items would not be: one could
//      belong to a CU still in an earlier task that waits for a task another CU
//      runs after this one.)
//   4. the CU's counter += 1 (task_post): allreduce_send waits for all NSM
// Weights in the gfx950 tile layout (amd/tools/shuffle_weights.py): W13 [NE]
// [2 IR / 16][LAT / 128][64][16 B] in item order, scales [NE][2 IR / 16][LAT /
// 512][64] dwords; W2 [NE][LAT / 16][IR / 128][64][16 B], scales [NE][LAT / 64]
// [64][4 row tiles x IR / 128] bytes (a lane's scales of 4 consecutive row
// tiles contiguous: 3 dwords per 4-tile item). The same byte counts as the
// B300 packing, so the graph's tensor shapes are the same on both backends. Map slots (as on CUDA): 0 W13, 2 W13 scales, 4 W2, 6 W2 scales
// (the others unused on gfx950). Buffer slots: 0 HQ, 1 HSF, 2 the work queues (the node's scratch).
#pragma once
#include "../core.cuh"
#include "sum_quant_send.cuh" // the layout of its input z_q
#include "topk_route.cuh"     // a local route (params[3] = 1): topk_route_task<.., true>

namespace static_mk {

// STATIC_MOE_AITER 1 (default 0): the expert weights and scales in AITER's layout (SGLang's process_weights_after_loading on
// K3's a16w4 path, aiter shuffle_weight_a16w4 / shuffle_scale_a16w4), so the server keeps one copy that SGLang's prefill uses
// too. The same bytes in another order (MI355X 10-24, CPU check box agent_tmp/aiterlay/layout_check.py): W2 weights the same;
// W13 row tiles 1 and 2 of each 4-tile item swapped (AITER: the item's 2 gate tiles first); W13 scales [NE][IR / 16][LAT / 256]
// [64] dwords, byte (ks % 2) 2 + (0 gate / 1 up) of the item's 16-feature tile pair; W2 scales [NE][LAT / 32][2][64] dwords
// (k-steps padded IR / 128 -> 4), byte (ks % 2) 2 + row tile % 2
#ifndef STATIC_MOE_AITER
#define STATIC_MOE_AITER 0
#endif
// the stored W13 row tile of an item's local tile tl (0 .. 3)
__host__ __device__ constexpr int moe_w13_tile(int tl) { return STATIC_MOE_AITER ? (tl == 1 ? 2 : (tl == 2 ? 1 : tl)) : tl; }
// AITER: the first of the 2 dwords (k-pairs 2 g, 2 g + 1, 64 dwords apart) that hold our scale group g (k-steps 4 g .. + 3) of
// item it's local tile tl, lane 0 (the caller adds the lane)
template <class C>
__device__ __forceinline__ unsigned const *moe_w13s_aiter(unsigned const *base, int e, int it, int tl, int g) {
  return base + (((size_t)e * (C::RT1 / 2) + 2 * it + (moe_w13_tile(tl) >> 1)) * (C::KS1 / 2) + 2 * g) * 64;
}
// AITER: our scale dword (byte j = k-step 4 g + j) from those 2 dwords, gu = the tile's gate (0) / up (1)
__device__ __forceinline__ unsigned moe_w13s_pack(unsigned a0, unsigned a1, int gu) {
  int const s = 8 * gu;
  return ((a0 >> s) & 0xFFu) | (((a0 >> (16 + s)) & 0xFFu) << 8) | (((a1 >> s) & 0xFFu) << 16) | (((a1 >> (16 + s)) & 0xFFu) << 24);
}
// a W2 weight scale: local row tile tl (0 .. 3) of a 4-tile group's dwords g4, k-step kq
template <class C>
__device__ __forceinline__ int moe_w2_sbyte(unsigned const *g4, int tl, int kq) {
#if STATIC_MOE_AITER
  return (g4[(tl >> 1) * 2 + (kq >> 1)] >> (8 * ((kq & 1) * 2 + (tl & 1)))) & 0xFF;
#else
  return (g4[(C::KS2 * tl + kq) / 4] >> (8 * ((C::KS2 * tl + kq) % 4))) & 0xFF;
#endif
}

// The task's choices (MI355X 10-11..13, measured in amd/notes/layer_math_mi355x.md):
// - W2 items of 8 row tiles (solo e2e 106.6 -> 97.4 us vs 4: twice the weight bytes in flight per wave)
// - a wave's W2 step: issue the next item's loads (weights, then h_q), compute and store the current item, then check
//   the next item's h_q (expert_w2_ready): loads and stores complete in issue order, so the check waits only for the
//   next item's loads, not for the previous item's R stores
// - the start in one round trip: z_q's scale rows are loaded with its pieces, and each thread's routing pair is loaded
//   before them (expert_table polls it only if it had not landed)
// - one counter per die hands out the die's W13 units, then its W2 items (tickets [0, U) W13 unit, [U, U + N2) W2
//   item): the ticket a wave takes for its next W13 unit (inside its unit) names its first W2 item once the die's W13
//   units are all taken, so it goes on without a round trip (MI355X 10-12, TP8 trace: 120-190 waves at 42-52 us
//   waiting for the W2 ticket after their last W13 unit). W13 units from the own die only; W2 items from another die
//   once its W13 units are all taken (read on its counter, expert_w13w2_take), so every W2 item's wait ends as before.
// the pause between two h_q polls of a W2 item, s_sleep(1) (a poll re-loads the slot's h_q, ~6 KB per wave)
// STATIC_W2_ONE 1 (W13CU builds, T <= 4): a wave holds one W2 item at a time and claims the next once the current one's h_q
// has landed (its ticket in flight during the compute), instead of two in flight with the next claimed before the current's
// loads (MI355X dec58, T = 1: the 64 CUs with no W13 unit took their W2 items at ~22 us, each claimed a second one while
// its first waited for h_q (~25), and ran it after the first: W2 items ended 36 (median) .. 45 us)
#ifndef STATIC_W2_ONE
#define STATIC_W2_ONE 0
#endif
// STATIC_MOE_ILV E (default 0; W13CU builds): each die's queue hands out its experts' W13 units and
// W2 items interleaved by groups of E experts - W13 (group 0), W13 (group 1), W2 (group 0), W13 (group 2), W2 (group 1), ..
// - a W2 ticket = 4 items of the group's expert (one per wave): the W2 items of the first experts run while the later
// experts' W13 units do, their h_q landed by then (MI355X 10-22 dec101, T = 4: W2 items only after every W13 unit of the
// die was taken, ~16 us of the experts phase W2 waiting). Every die finishes its own W2 items (no W2 from other dies).
#ifndef STATIC_MOE_ILV
#define STATIC_MOE_ILV 0
#endif
// STATIC_MOE_ILVW E (default 0; builds with a W13 item per wave, not W13CU): the same interleave for
// per-wave tickets - each wave takes an entry of its die's sequence, a W13 item or a W2 item (the next ticket issued after the
// current entry's loads)
#ifndef STATIC_MOE_ILVW
#define STATIC_MOE_ILVW 0
#endif
// STATIC_FAST_TABLE 1: the routing table of T = 8's (no serial chains, a presence bitmask) at T = 1, 2, 4, 16 too
#ifndef STATIC_FAST_TABLE
#define STATIC_FAST_TABLE 0
#endif
// STATIC_MOE_W2_TILES N (default 8): row tiles per W2 item (the R block counts stay per 8 tiles)
#ifndef STATIC_MOE_W2_TILES
#define STATIC_MOE_W2_TILES 8
#endif
constexpr int W13_WS_TILES = 4; // row tiles of a wave's W13 scales in LDS
__device__ __forceinline__ void w2_poll_pause() {
  __builtin_amdgcn_s_sleep(1);
  asm volatile("" ::: "memory");
}

// the weight fragments' loads: nt (non-temporal: amd/bench/expert_unit.hip 10-10, W13 3.3 vs 2.0 TB/s with plain loads)
__device__ __forceinline__ v4i wload(v4i const *p) {
  return __builtin_nontemporal_load(p);
}

// where feature k (0..127) of a K step sits in an fp8 operand: lane group g and
// byte b of the lane's 32 (the ISA's layout: K 16g..16g+15 and 64+16g..64+16g+15)
__device__ __forceinline__ void fp8_slot(int k, int &g, int &b) {
  if (k < 64) {
    g = k / 16;
    b = k % 16;
  } else {
    g = (k - 64) / 16;
    b = 16 + (k - 64) % 16;
  }
}

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
struct ExpertCfg {
  static constexpr int NE = NE_, K = K_, LAT = LAT_, IR = IR_;
  // the map slots (Maps::m) of the weights and their scales
  static constexpr int W13M = W13M_, W13SFM = W13SFM_, W2M = W2M_,
                       W2SFM = W2SFM_;
  static constexpr int NPAIR = T * K, NSLOT = T * K;
  // counters: CNT the CUs done; CHQ[slot] the CUDA template's W13 items done
  // (not used on gfx950: W2 polls h_q itself; CNT + 32: the CUDA template's
  // queue, not used either: the queues are buffer slot 2)
  static constexpr int CNT = CNT_, CHQ = CNT_ + 64;
  static constexpr size_t ZOFF = ZOFF_;
  static constexpr int KS1 = LAT / 128, KG1 = KS1 / 4, RT1 = 2 * IR / 16,
                       ITEMS1 = IR / 32;
  // the misc scratch (static_megakernel.moe_experts_layer): the queues, then
  // 8 KB not used (a count per W13 item of the tail split, removed), then the trace
  static constexpr size_t MISC_SPLIT = 16 * 128, MISC_TRACE = 16 * 128 + 8192;
  static constexpr int KS2 = IR / 128, RT2 = LAT / 16,
                       W2_TILES = STATIC_MOE_W2_TILES, ITEMS2 = RT2 / W2_TILES;
  static constexpr int STAGES = 7;
  static_assert(LAT % 512 == 0 && IR % 128 == 0 && IR / 128 <= 4,
                "moe_experts: whole scale groups of W13, one scale dword per W2 "
                "tile");
  static_assert(RT2 % W2_TILES == 0 && W2_TILES % 4 == 0,
                "moe_experts: whole W2 items of whole 4-tile scale groups");
  static constexpr int SCW = STATIC_MOE_AITER ? 4 : KS2; // scale dwords per 4 tiles (4 x KS2 bytes; AITER: 2 x 2 x 2 k-steps)
  static_assert(KS1 % STAGES == 0 && KS1 > STAGES,
                "moe_experts: whole rounds of W13 stages");
  static_assert(NE % 64 == 0 && NE / 64 <= 16 && NPAIR <= 256,
                "moe_experts: the table's 64-expert blocks, a thread per pair");
  static_assert(T <= 16, "moe_experts: the tokens are the MFMA's 16 columns");
  static_assert(CHQ + NSLOT <= CNT + 32 * NODE_COUNTER_LINES,
                "moe_experts: CHQ in the node's counter lines");
};

// the task's LDS (from the 1024-aligned dynamic base)
template <class C>
struct ExpertLds {
  static constexpr int Z = 0;                         // [KS1][64][32] fp8
  static constexpr int ZS = Z + C::KS1 * 64 * 32;     // [KS1][64] e8m0
  static constexpr int WS = ZS + C::KS1 * 64;         // [4 waves][4 (W13_PAIR: 8)][KG1][64] u32
  static constexpr int CNT_E = WS + 4 * W13_WS_TILES * C::KG1 * 64 * 4; // [NE] int
  static constexpr int SEL = CNT_E + C::NE * 4;       // [NPAIR] int
  static constexpr int WSEL = SEL + C::NPAIR * 4;     // [NPAIR] float
  static constexpr int EID = WSEL + C::NPAIR * 4;     // [NSLOT] int
  static constexpr int NTOK = EID + C::NSLOT * 4;     // [NSLOT] int
  static constexpr int TOK = NTOK + C::NSLOT * 4;     // [NSLOT][T] int
  static constexpr int WT = TOK + C::NSLOT * T * 4;   // [NSLOT][T] float
  static constexpr int KK = WT + C::NSLOT * T * 4;    // [NSLOT][T] u8
  static constexpr int POS = KK + C::NSLOT * T;       // [NSLOT][T] s8
  static constexpr int BLK = POS + C::NSLOT * T;      // [16] int + n_live
  // (CUQ, XCH: not used; kept, so the task's LDS size, and the launch's, stays as measured)
  static constexpr int CUQ = BLK + 17 * 4;
  static constexpr int XCH = (CUQ + 4 + 15) / 16 * 16;
  static constexpr int WW = XCH + 4 * 64 * 16;       // (4 waves x 256 B, unused; kept, as CUQ)
  static constexpr int BYTES = WW;
};

// 1. z_q -> LDS fragments
template <class C>
__device__ __forceinline__ void expert_load_z(G const &g, char *sm) {
  using Ls = ExpertLds<C>;
  unsigned char *z = reinterpret_cast<unsigned char *>(sm + Ls::Z),
                *zs = reinterpret_cast<unsigned char *>(sm + Ls::ZS);
  // the lanes of columns T..15 stay 0 (the MFMA's unused token columns)
  for (int i = STATIC_TIDX; i < C::KS1 * 64 * 32 / 16; i += 256) {
    reinterpret_cast<uint4 *>(z)[i] = make_uint4(0, 0, 0, 0);
  }
  for (int i = STATIC_TIDX; i < C::KS1 * 64; i += 256) {
    zs[i] = 127;
  }
  __syncthreads();
  // z_q bytes, gathered 16 B at a time: in both fp8 layouts each half of a
  // lane's 32-byte fragment is 16 consecutive K values of one token, so piece
  // v = (K step ks, lane group gq, half h, token t) is one 16-B copy from
  // z_q[t][128 ks + k0] (k0 = 64 h + 16 gq). All
  // of this thread's pieces are polled at once (system scope, tracked loads),
  // re-loaded until all landed. (A byte-by-byte scatter made the compiler hoist
  // ~100 per-byte addresses out of the task loop and spill.)
  // a cheap wait first: per sending GPU one 16-B piece, the scale row of the
  // last z block it quantizes (sum_quant_send pushes a block's scales after
  // its bytes), polled with a longer sleep until every one has landed; then the
  // full poll below finds nearly every piece there (MI355X 10-11, TP8: each CU
  // re-loading all of its 1792 pieces every round while the other GPUs' z had
  // not landed, ~108 CUs from ~2 us on, put TB/s of system-scope loads on the
  // memory, slowing the GEMMs then running and the z writes themselves)
  uint32_t const zoff = (uint32_t)(C::ZOFF + rg_set());
  constexpr int NZ = C::KS1 * 4 * 2 * T, PER = (NZ + 255) / 256;
  auto piece = [](int v, int &dst, uint32_t &src) {
    int const t = v % T, h = (v / T) % 2, gq = (v / (2 * T)) % 4,
              ks = v / (8 * T);
    int const k0 = 64 * h + 16 * gq;
    dst = ((ks * 64 + 16 * gq + t) * 32) + 16 * h;
    src = (uint32_t)(t * C::LAT + 128 * ks + k0);
  };
  uint4 u[PER];
  bool ok[PER];
  // SOLO: the K steps other GPUs quantize (sum_quant_send's blocks, divided
  // over the GPUs as compiler.owned_row_blocks) stay zeros
  int const ks_lo = C::KS1 * g.rank / GPUS, ks_hi = C::KS1 * (g.rank + 1) / GPUS;
  // the scale rows (chunk ks, row t) in the same polls
  constexpr int NS = C::KS1 * T, PS = (NS + 255) / 256;
  uint4 us[PS];
  bool oks[PS];
#pragma unroll
  for (int q = 0; q < PS; q++) {
    int const v = STATIC_TIDX + 256 * q, ks = v / T;
    oks[q] = v >= NS || (SOLO && (ks < ks_lo || ks >= ks_hi));
    us[q] = make_uint4(0, 0, 0, 0);
  }
#pragma unroll
  for (int q = 0; q < PER; q++) {
    int const v = STATIC_TIDX + 256 * q, ks = v / (8 * T);
    ok[q] = v >= NZ || (SOLO && (ks < ks_lo || ks >= ks_hi));
    u[q] = make_uint4(0, 0, 0, 0);
  }
  bool all;
  do {
#pragma unroll
    for (int q = 0; q < PER; q++) {
      if (!ok[q]) {
        int dst;
        uint32_t so;
        piece(STATIC_TIDX + 256 * q, dst, so);
        u[q] = ld16_at<AUX_SYS>(g.rv, zoff + so);
      }
    }
#pragma unroll
    for (int q = 0; q < PS; q++) {
      if (!oks[q]) {
        int const v = STATIC_TIDX + 256 * q, ks = v / T, t = v % T;
        us[q] = ld16_at<AUX_SYS>(
            g.rv, zoff + (uint32_t)(zq_scales_at(C::LAT) + (size_t)ks * ZQ_CHUNK + t * 16));
      }
    }
    all = true;
#pragma unroll
    for (int q = 0; q < PER; q++) {
      ok[q] = ok[q] || valid16w(u[q]);
      all = all && ok[q];
    }
#pragma unroll
    for (int q = 0; q < PS; q++) {
      oks[q] = oks[q] || valid16w(us[q]);
      all = all && oks[q];
    }
    if (!all) {
      poll_pause();
    }
  } while (!all);
#pragma unroll
  for (int q = 0; q < PER; q++) {
    if (STATIC_TIDX + 256 * q < NZ) {
      int dst;
      uint32_t so;
      piece(STATIC_TIDX + 256 * q, dst, so);
      *reinterpret_cast<uint4 *>(z + dst) = u[q];
    }
  }
#pragma unroll
  for (int q = 0; q < PS; q++) {
    int const v = STATIC_TIDX + 256 * q, ks = v / T, t = v % T;
    if (v < NS && !(SOLO && (ks < ks_lo || ks >= ks_hi))) {
#pragma unroll
      for (int kb = 0; kb < 4; kb++) {
        zs[ks * 64 + 16 * kb + t] = (unsigned char)(us[q].x >> (8 * kb));
      }
    }
  }
}

// 2. the routing table
template <class C>
__device__ __forceinline__ int
    expert_table(unsigned long long const *pairs, char *sm,
                 unsigned long long pre = ~0ull) {
  using Ls = ExpertLds<C>;
  int *cnt_e = reinterpret_cast<int *>(sm + Ls::CNT_E),
      *sel = reinterpret_cast<int *>(sm + Ls::SEL),
      *eid = reinterpret_cast<int *>(sm + Ls::EID),
      *ntok = reinterpret_cast<int *>(sm + Ls::NTOK),
      *tok = reinterpret_cast<int *>(sm + Ls::TOK),
      *blk = reinterpret_cast<int *>(sm + Ls::BLK);
  float *wsel = reinterpret_cast<float *>(sm + Ls::WSEL),
        *wt = reinterpret_cast<float *>(sm + Ls::WT);
  unsigned char *kk = reinterpret_cast<unsigned char *>(sm + Ls::KK);
  signed char *pos = reinterpret_cast<signed char *>(sm + Ls::POS);
  constexpr bool FAST = (C::NPAIR == 128 && C::NSLOT * T == 1024) || // T = 8; T 1, 2, 4, 16 with STATIC_FAST_TABLE (dec63
                        (STATIC_FAST_TABLE && C::NPAIR <= 256 && (C::NSLOT * T) % 4 == 0); // T = 16: the general table 4.0 us)
  if constexpr (FAST) {
    // The table with no serial chains and a presence bitmask (MI355X 10-16: z_q -> table 3.2-3.8 us
    // before; with a list of each expert's first pair, its reading was 1.6 of the table's 2.3 us): A the pair's expert,
    // cnt_e[e] and the NE / 32 presence words (in the W13 scales' staging, free until the first W13 unit) zeroed, pos to
    // -1; B the token bit into cnt_e[e], the expert's bit into its word; C slot = the bits below e (NE / 128 16-B reads),
    // n_live = all bits, the position = the earlier tokens that picked e
    static_assert(C::NPAIR <= 256 && (C::NSLOT * T) % 4 == 0 && C::NE % 128 == 0 && C::NE / 32 <= 128,
                  "expert_table: at most 256 pairs, pos as words, presence words in the W13 scales' staging");
    constexpr int NW = C::NE / 32, PW = C::NSLOT * T / 4; // presence words; pos as words
    unsigned *const pres = reinterpret_cast<unsigned *>(sm + Ls::WS);
    if constexpr (PW == 256) {
      reinterpret_cast<int *>(pos)[STATIC_TIDX] = -1; // (4 x int8 -1)
    } else {
      for (int i = STATIC_TIDX; i < PW; i += 256) {
        reinterpret_cast<int *>(pos)[i] = -1;
      }
    }
    if (STATIC_TIDX >= 256 - NW) {
      pres[STATIC_TIDX - (256 - NW)] = 0u;
    }
    int e = 0;
    if (STATIC_TIDX < C::NPAIR) {
      unsigned long long v = pre;
      while (v == ~0ull) {
        v = ld64_dev(pairs + STATIC_TIDX);
        if (v == ~0ull) {
          __builtin_amdgcn_s_sleep(1);
        }
      }
      e = (int)(v >> 32);
      sel[STATIC_TIDX] = e;
      wsel[STATIC_TIDX] = __uint_as_float((uint32_t)v);
      cnt_e[e] = 0;
    }
    __syncthreads();
    if (STATIC_TIDX < C::NPAIR) {
      atomicOr(&cnt_e[e], 1 << (STATIC_TIDX / C::K));
      atomicOr(&pres[e >> 5], 1u << (e & 31));
    }
    __syncthreads();
    int below = 0, n_live = 0;
    int const we = e >> 5;
#pragma unroll
    for (int w4 = 0; w4 < NW; w4 += 4) {
      uint4 const p4 = *reinterpret_cast<uint4 const *>(pres + w4);
      unsigned const pw[4] = {p4.x, p4.y, p4.z, p4.w};
#pragma unroll
      for (int q = 0; q < 4; q++) {
        int const w = w4 + q;
        n_live += __popc(pw[q]);
        below += w < we ? __popc(pw[q]) : (w == we ? __popc(pw[q] & ((1u << (e & 31)) - 1u)) : 0);
      }
    }
    if (STATIC_TIDX < C::NPAIR) {
      int const tmask = cnt_e[e];
      int const t = STATIC_TIDX / C::K, lo = below, p = __popc(tmask & ((1 << t) - 1));
      if (p == 0) {
        eid[lo] = e;
        ntok[lo] = __popc(tmask);
      }
      tok[lo * T + p] = t;
      wt[lo * T + p] = wsel[STATIC_TIDX];
      kk[lo * T + p] = (unsigned char)(STATIC_TIDX % C::K);
      pos[lo * T + t] = (signed char)p;
    }
    __syncthreads();
    (void)blk;
    return n_live;
  } else {
    // the general table (T != 8, at most 256 pairs: one thread per pair): slots in ascending expert id by 64-lane ballots
    static_assert(C::NPAIR <= 256 && T <= 16, "expert_table: one thread per pair, a token bit mask per expert");
    for (int i = STATIC_TIDX; i < C::NE; i += 256) {
      cnt_e[i] = 0;
    }
    for (int i = STATIC_TIDX; i < C::NSLOT * T; i += 256) {
      pos[i] = -1;
    }
    __syncthreads();
    if (STATIC_TIDX < C::NPAIR) { // poll pair i (0xFF-prefilled; loaded at the task's start)
      unsigned long long v = pre;
      while (v == ~0ull) {
        v = ld64_dev(pairs + STATIC_TIDX);
        if (v == ~0ull) {
          __builtin_amdgcn_s_sleep(1);
        }
      }
      int const e = (int)(v >> 32);
      sel[STATIC_TIDX] = e;
      wsel[STATIC_TIDX] = __uint_as_float((uint32_t)v);
      // cnt_e[e]: a bit per token that picked e (an OR: the same table on every CU, whatever order the CUs' atomics take)
      atomicOr(&cnt_e[e], 1 << (STATIC_TIDX / C::K));
    }
    __syncthreads();
    // slots in ascending expert id: 64-expert blocks, wave w takes blocks w, w + 4, ...; a block's live experts by one
    // 64-lane ballot
    int const wave = STATIC_TIDX >> 6, lane = STATIC_TIDX & 63;
    constexpr int NB = C::NE / 64;
    unsigned long long m[4] = {};
#pragma unroll
    for (int q = 0; q < 4; q++) {
      int const b = wave + 4 * q;
      bool const live = b < NB && cnt_e[b * 64 + lane] > 0;
      m[q] = __ballot(live);
      if (b < NB && lane == 0) {
        blk[b] = __popcll(m[q]);
      }
    }
    __syncthreads();
#pragma unroll
    for (int q = 0; q < 4; q++) {
      int const b = wave + 4 * q;
      if (b < NB && cnt_e[b * 64 + lane] > 0) {
        int base = 0;
        for (int b_ = 0; b_ < b; b_++) {
          base += blk[b_];
        }
        int const slot = base + __popcll(m[q] & ((1ull << lane) - 1ull));
        eid[slot] = b * 64 + lane;
        ntok[slot] = 0;
      }
    }
    if (STATIC_TIDX == 0) {
      int n = 0;
      for (int b_ = 0; b_ < NB; b_++) {
        n += blk[b_];
      }
      blk[16] = n;
    }
    __syncthreads();
    int const n_live = blk[16];
    if (STATIC_TIDX < C::NPAIR) { // pair -> its slot (binary search), position
      int const t = STATIC_TIDX / C::K, e = sel[STATIC_TIDX];
      int const tmask = cnt_e[e];
      int lo = 0, hi = n_live - 1;
      while (lo < hi) {
        int const mid = (lo + hi) >> 1;
        if (eid[mid] < e) {
          lo = mid + 1;
        } else {
          hi = mid;
        }
      }
      // the token's position in the slot: the earlier tokens that picked e (a token picks an expert at most once; every
      // CU builds the same table, so the CUs of a slot's W13 and W2 items agree on the columns)
      int const p = __popc(tmask & ((1 << t) - 1));
      if (p == 0) {
        ntok[lo] = __popc(tmask);
      }
      tok[lo * T + p] = t;
      wt[lo * T + p] = wsel[STATIC_TIDX];
      kk[lo * T + p] = (unsigned char)(STATIC_TIDX % C::K);
      pos[lo * T + t] = (signed char)p;
    }
    __syncthreads();
    return n_live;
  }
}

// 3a. one W13 item of one wave: slot's expert e, features [32 it, 32 it + 32)
// a W13 item's row tiles T0 .. T0 + NTW - 1 (of its 4: 0, 1 gate, 2, 3 up)
// over K steps KS0 .. KS0 + NKS - 1 (whole scale groups of 4 steps), STG K
// steps of weights in flight -> acc[NTW]
// (as two parts: expert_w13_issue, the scale loads and the first STG K steps'
// weight loads into pf / ps, flat [STG][NTW] / [NTW][NG], at most W13_PF / W13_PS
// entries; expert_w13_run, the rest)
constexpr int W13_PF = 32, W13_PS = 32;
// expert_w13_run calls its hook once, after the unit's last weight loads are
// issued, before the last round's MFMAs (the next unit's ticket)
struct W13NoHook {
  __device__ void operator()() const {}
};
template <class C, int T0, int NTW, int STG, int KS0 = 0, int NKS = C::KS1>
__device__ __forceinline__ void expert_w13_issue(Maps const &maps,
                                                 char const *sm,
                                                 int slot,
                                                 int it,
                                                 v4i *pf,
                                                 unsigned *ps) {
  using Ls = ExpertLds<C>;
  static_assert(NKS % STG == 0 && NKS >= STG, "whole rounds of STG");
  static_assert(KS0 % 4 == 0 && NKS % 4 == 0 && KS0 + NKS <= C::KS1,
                "whole scale groups of 4 K steps");
  constexpr int G0 = KS0 / 4, NG = NKS / 4;
  static_assert(STG * NTW <= W13_PF && NTW * NG <= W13_PS, "W13_PF / W13_PS");
  int const lane = STATIC_TIDX & 63;
  int const e = reinterpret_cast<int const *>(sm + Ls::EID)[slot];
  v4i const *wt = reinterpret_cast<v4i const *>(maps.m[C::W13M].p) +
                  ((size_t)e * C::RT1 + 4 * it + T0) * C::KS1 * 64 + lane;
  unsigned const *st = reinterpret_cast<unsigned const *>(maps.m[C::W13SFM].p) +
                       ((size_t)e * C::RT1 + 4 * it + T0) * C::KG1 * 64 + lane;
  // the item's weight scales first (vmcnt counts in order: a scale load inside
  // the K loop would make its use wait for every weight load before it): all
  // their loads, then the first STG K steps' weight loads, then the scales' LDS
  // stores (written load, store, load, store, the compiler waited for each
  // pair: MI355X 10-10, 14 round trips per item before any weight load)
#if STATIC_MOE_AITER
  unsigned raw[2 * NTW * NG]; // (AITER: 2 dwords per scale group, packed once the first weight loads are issued)
  unsigned const *sa = reinterpret_cast<unsigned const *>(maps.m[C::W13SFM].p) + lane;
#pragma unroll
  for (int t = 0; t < NTW; t++)
#pragma unroll
    for (int g = 0; g < NG; g++) {
      unsigned const *p = moe_w13s_aiter<C>(sa, e, it, T0 + t, G0 + g);
      raw[2 * (t * NG + g)] = __builtin_nontemporal_load(p);
      raw[2 * (t * NG + g) + 1] = __builtin_nontemporal_load(p + 64);
    }
  (void)st;
#else
#pragma unroll
  for (int t = 0; t < NTW; t++)
#pragma unroll
    for (int g = 0; g < NG; g++) {
      ps[t * NG + g] =
          __builtin_nontemporal_load(st + ((size_t)t * C::KG1 + G0 + g) * 64);
    }
#endif
#pragma unroll
  for (int s = 0; s < STG; s++)
#pragma unroll
    for (int t = 0; t < NTW; t++) {
#if STATIC_MOE_AITER
      pf[s * NTW + t] = wload(wt + ((ptrdiff_t)(moe_w13_tile(T0 + t) - T0) * C::KS1 + KS0 + s) * 64);
#else
      pf[s * NTW + t] = wload(wt + ((size_t)t * C::KS1 + KS0 + s) * 64);
#endif
    }
#if STATIC_MOE_AITER
  asm volatile("" ::: "memory");
#pragma unroll
  for (int t = 0; t < NTW; t++)
#pragma unroll
    for (int g = 0; g < NG; g++) {
      ps[t * NG + g] = moe_w13s_pack(raw[2 * (t * NG + g)], raw[2 * (t * NG + g) + 1], moe_w13_tile(T0 + t) & 1);
    }
#endif
}
template <class C, int T0, int NTW, int STG, int KS0 = 0, int NKS = C::KS1,
          class Hook = W13NoHook>
__device__ __forceinline__ void expert_w13_run(Maps const &maps,
                                               char *sm,
                                               int slot,
                                               int it,
                                               v4i const *pf,
                                               unsigned const *ps,
                                               v4f *acc,
                                               Hook const &hook = Hook{}) {
  using Ls = ExpertLds<C>;
  constexpr int G0 = KS0 / 4, NG = NKS / 4;
  int const lane = STATIC_TIDX & 63, wave = STATIC_TIDX >> 6;
  int const e = reinterpret_cast<int const *>(sm + Ls::EID)[slot];
  v8i const *z_l = reinterpret_cast<v8i const *>(sm + Ls::Z);
  unsigned char const *zs_l =
      reinterpret_cast<unsigned char const *>(sm + Ls::ZS);
  unsigned *ws_l = reinterpret_cast<unsigned *>(sm + Ls::WS) +
                   wave * W13_WS_TILES * C::KG1 * 64; // [NTW][KG1][64] of [4 (W13_PAIR: 8)][KG1][64]
  static_assert(T0 + NTW <= W13_WS_TILES, "expert_w13_run: the wave's scale tiles in LDS");
  v4i const *wt = reinterpret_cast<v4i const *>(maps.m[C::W13M].p) +
                  ((size_t)e * C::RT1 + 4 * it + T0) * C::KS1 * 64 + lane;
#pragma unroll
  for (int t = 0; t < NTW; t++) {
    acc[t] = v4f{0.f, 0.f, 0.f, 0.f};
  }
  v4i frag[STG][NTW];
#pragma unroll
  for (int s = 0; s < STG; s++)
#pragma unroll
    for (int t = 0; t < NTW; t++) {
      frag[s][t] = pf[s * NTW + t];
    }
  // every load above is issued before the scales' LDS stores, which wait only
  // for the (older) scale loads
  asm volatile("" ::: "memory");
#pragma unroll
  for (int t = 0; t < NTW; t++)
#pragma unroll
    for (int g = 0; g < NG; g++) {
      ws_l[(t * C::KG1 + G0 + g) * 64 + lane] = ps[t * NG + g];
    }
  auto step = [&](int ks, v4i const *f) {
    v8i const b = z_l[ks * 64 + lane];
    int const bs = zs_l[ks * 64 + lane];
#pragma unroll
    for (int t = 0; t < NTW; t++) {
      v8i const a = {f[t][0], f[t][1], f[t][2], f[t][3], 0, 0, 0, 0};
      int const as = (ws_l[(t * C::KG1 + ks / 4) * 64 + lane] >> (8 * (ks % 4))) & 0xFF;
      acc[t] = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
          a, b, acc[t], 4 /*A fp4*/, 0 /*B fp8*/, 0, as, 0, bs);
    }
  };
  // rounds of STG K steps; the last round peeled (no branch around the
  // prefetch in the others)
#pragma unroll 1
  for (int k0 = KS0; k0 < KS0 + NKS - STG; k0 += STG) {
#pragma unroll
    for (int s = 0; s < STG; s++) {
      step(k0 + s, frag[s]);
#pragma unroll
      for (int t = 0; t < NTW; t++) {
#if STATIC_MOE_AITER
        frag[s][t] = wload(wt + ((ptrdiff_t)(moe_w13_tile(T0 + t) - T0) * C::KS1 + k0 + s + STG) * 64);
#else
        frag[s][t] = wload(wt + ((size_t)t * C::KS1 + k0 + s + STG) * 64);
#endif
      }
    }
  }
  hook();
#pragma unroll
  for (int s = 0; s < STG; s++) {
    step(KS0 + NKS - STG + s, frag[s]);
  }
}

template <class C, int T0, int NTW, int STG, int KS0 = 0, int NKS = C::KS1,
          class Hook = W13NoHook>
__device__ __forceinline__ void expert_w13_acc(Maps const &maps,
                                               char *sm,
                                               int slot,
                                               int it,
                                               v4f *acc,
                                               Hook const &hook = Hook{}) {
  v4i pf[W13_PF];
  unsigned ps[W13_PS];
  expert_w13_issue<C, T0, NTW, STG, KS0, NKS>(maps, sm, slot, it, pf, ps);
  expert_w13_run<C, T0, NTW, STG, KS0, NKS>(maps, sm, slot, it, pf, ps, acc, hook);
}

// the item's h: SiTU(gate, up) (gate: tiles 0, 1; up: tiles 2, 3) -> MXFP8 h_q
// of the slot's tokens
template <class C>
__device__ __forceinline__ void expert_w13_finish(char *sm,
                                                  int slot,
                                                  int it,
                                                  v4f const *gate,
                                                  v4f const *up,
                                                  unsigned char *hq,
                                                  unsigned char *hsc) {
  using Ls = ExpertLds<C>;
  int const lane = STATIC_TIDX & 63;
  // this lane: token l % 16, features 16 half + 4 (l / 16) + r
  int const tk = lane % 16;
  float hv[2][4], amax = 0.f;
#pragma unroll
  for (int half = 0; half < 2; half++)
#pragma unroll
    for (int r = 0; r < 4; r++) {
      hv[half][r] = situ(gate[half][r], up[half][r]);
      amax = fmaxf(amax, fabsf(hv[half][r]));
    }
  amax = fmaxf(amax, __shfl_xor(amax, 16)); // the token's 32 features: lanes
  amax = fmaxf(amax, __shfl_xor(amax, 32)); // tk, tk + 16, tk + 32, tk + 48
  int ex = ceil_log2_bits(fmaxf(amax, 1.0e-30f) * (1.0f / 448.0f));
  ex = ex < -127 ? -127 : (ex > 127 ? 127 : ex);
  float const inv = __uint_as_float((uint32_t)(127 - ex) << 23);
  // h_q of the slot's tokens, column p = the token's position in the slot
  int const p = tk < T ? reinterpret_cast<signed char const *>(
                             sm + Ls::POS)[slot * T + tk]
                       : -1;
  if (p >= 0) {
    int const ks2 = it / 4, q = it % 4; // W2's K step, 32-feature block
    unsigned char *hq_s = hq + (size_t)slot * C::KS2 * 4 * T * 32;
#pragma unroll
    for (int half = 0; half < 2; half++)
#pragma unroll
      for (int r = 0; r < 4; r++) {
        int const f = 16 * half + 4 * (lane / 16) + r;
        int gq, b;
        fp8_slot(32 * q + f, gq, b);
        st8_out(hq_s + ((size_t)ks2 * 4 * T + T * gq + p) * 32 + b,
                cvt_e4m3(hv[half][r] * inv));
      }
    if (lane < 16) {
      st8_out(hsc + ((size_t)slot * C::KS2 + ks2) * 64 + 16 * q + p,
              (uint8_t)(ex + 127));
    }
  }
}

// 3a'. a W13 item shared by a CU's 4 waves (params[1] = 1: few items, T <= 4): wave w computes the item's row tile w over
// the whole K, every K step's weight load in flight at once (as many registers as the per-wave unit's 7 steps x 4 tiles);
// the 4 sums meet in LDS (the per-wave unit's sums, the same bits) and wave 0 makes h_q
template <class C>
struct W13Tile {
  v4i f[C::KS1];
#if STATIC_MOE_AITER
  unsigned sc[C::KS1 / 2]; // (AITER: k-pair dwords, byte (ks % 2) 2 + gu)
  int gu;
#else
  unsigned sc[C::KG1];
#endif
};
template <class C>
__device__ __forceinline__ void w13_tile_issue(W13Tile<C> &F, Maps const &maps, char const *sm, int slot, int it,
                                               int tile) {
  int const lane = STATIC_TIDX & 63;
  int const e = reinterpret_cast<int const *>(sm + ExpertLds<C>::EID)[slot];
#if STATIC_MOE_AITER
  int const ta = tile == 1 ? 2 : (tile == 2 ? 1 : tile);
  size_t const rt = (size_t)e * C::RT1 + 4 * it + ta;
  v4i const *wt = reinterpret_cast<v4i const *>(maps.m[C::W13M].p) + rt * C::KS1 * 64 + lane;
  unsigned const *st = reinterpret_cast<unsigned const *>(maps.m[C::W13SFM].p) +
                       (((size_t)e * (C::RT1 / 2) + 2 * it + (ta >> 1)) * (C::KS1 / 2)) * 64 + lane;
  F.gu = ta & 1;
#pragma unroll
  for (int g = 0; g < C::KS1 / 2; g++) {
    F.sc[g] = __builtin_nontemporal_load(st + (size_t)g * 64);
  }
#else
  size_t const rt = (size_t)e * C::RT1 + 4 * it + tile;
  v4i const *wt = reinterpret_cast<v4i const *>(maps.m[C::W13M].p) + rt * C::KS1 * 64 + lane;
  unsigned const *st = reinterpret_cast<unsigned const *>(maps.m[C::W13SFM].p) + rt * C::KG1 * 64 + lane;
#pragma unroll
  for (int g = 0; g < C::KG1; g++) {
    F.sc[g] = __builtin_nontemporal_load(st + (size_t)g * 64);
  }
#endif
#pragma unroll
  for (int s = 0; s < C::KS1; s++) {
    F.f[s] = wload(wt + (size_t)s * 64);
  }
}
template <class C>
__device__ __forceinline__ v4f w13_tile_compute(W13Tile<C> const &F, char const *sm) {
  using Ls = ExpertLds<C>;
  int const lane = STATIC_TIDX & 63;
  v8i const *z_l = reinterpret_cast<v8i const *>(sm + Ls::Z);
  unsigned char const *zs_l = reinterpret_cast<unsigned char const *>(sm + Ls::ZS);
  v4f acc = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int ks = 0; ks < C::KS1; ks++) {
    v8i const b = z_l[ks * 64 + lane];
    int const bs = zs_l[ks * 64 + lane];
    v8i const a = {F.f[ks][0], F.f[ks][1], F.f[ks][2], F.f[ks][3], 0, 0, 0, 0};
#if STATIC_MOE_AITER
    int const as = (F.sc[ks / 2] >> (8 * ((ks % 2) * 2 + F.gu))) & 0xFF;
#else
    int const as = (F.sc[ks / 4] >> (8 * (ks % 4))) & 0xFF;
#endif
    acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, acc, 4, 0, 0, as, 0, bs);
  }
  return acc;
}

// 3b. W2: an item's operands (the weights and scales, then the wait for h_q, then h_q)
template <class C>
struct W2Item {
  v8i b[C::KS2];
  int bs[C::KS2];
  v4i f[C::W2_TILES][C::KS2];
  unsigned sc[C::W2_TILES / 4][C::SCW]; // byte KS2 (tile % 4) + k
#ifdef STATIC_TIMING_BUILD
  long long t_wait = 0; // the timing build: ns waiting for h_q in w2_load
  long long t_load = 0; // the timing build: when its loads were issued
#endif
};
// the slot's h_q and its scales into F (live lanes; the others zeros). h_q is its own ready flag: h_q and its scales
// are 0xFF before each launch and no e4m3 byte (saturating) or e8m0 scale (ex + 127 <= 254) the W13 items write is
// 0xFF, so the loads are re-issued until every byte the lane uses is written; no counter round trip before them
// (MI355X 10-11: the W2 phase took ~28 us for ~60 MB, a counter poll then the h_q loads per item; with this, solo e2e
// 116 -> 104 us, W2 phase ~33 -> ~24 us). Only the slot's n token columns are used; the others are zeros.
template <class C>
__device__ __forceinline__ void expert_w2_hq(W2Item<C> &F,
                                             char const *sm,
                                             int slot,
                                             unsigned char const *hq,
                                             unsigned char const *hsc) {
  int const lane = STATIC_TIDX & 63, col = lane % 16, grp_l = lane / 16;
  int const n = reinterpret_cast<int const *>(sm + ExpertLds<C>::NTOK)[slot];
#pragma unroll
  for (int k = 0; k < C::KS2; k++) {
    F.b[k] = v8i{0, 0, 0, 0, 0, 0, 0, 0};
    F.bs[k] = 0;
    if (col < n) {
      uint32_t const hoff = (uint32_t)(
          (((size_t)slot * C::KS2 + k) * 4 * T + T * grp_l + col) * 32);
      uint4 const lo = ld16_in_at(hq, hoff), hi = ld16_in_at(hq, hoff + 16);
      F.b[k] = v8i{(int)lo.x, (int)lo.y, (int)lo.z, (int)lo.w,
                   (int)hi.x, (int)hi.y, (int)hi.z, (int)hi.w};
      F.bs[k] = ld8_in_at(
          hsc, (uint32_t)(((size_t)slot * C::KS2 + k) * 64 + lane));
    }
  }
}
template <class C>
__device__ __forceinline__ bool expert_w2_ok(W2Item<C> const &F,
                                             char const *sm,
                                             int slot) {
  int const lane = STATIC_TIDX & 63;
  int const n = reinterpret_cast<int const *>(sm + ExpertLds<C>::NTOK)[slot];
  auto ff_free = [](uint32_t w) { // no byte of w is 0xFF
    uint32_t const t = ~w;
    return ((t - 0x01010101u) & ~t & 0x80808080u) == 0u;
  };
  bool ok = true;
  if (lane % 16 < n) {
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
// STATIC_MOE_W2_HQ_FIRST 1 (default 0): a W2 item's h_q loads before its weights (as moe_experts_csr.cuh STATIC_CSR_W2_HQ_FIRST:
// MI355X 10-26 dlib60 T 16 lib.s: vmcnt(27) then vmcnt(0) before each item's first MFMA)
#ifndef STATIC_MOE_W2_HQ_FIRST
#define STATIC_MOE_W2_HQ_FIRST 0
#endif
template <class C>
__device__ __forceinline__ void expert_w2_load(W2Item<C> &F,
                                               Maps const &maps,
                                               G const &g,
                                               char const *sm,
                                               int slot,
                                               int t0,
                                               unsigned char const *hq,
                                               unsigned char const *hsc) {
#ifdef STATIC_TIMING_BUILD
  F.t_load = gtime();
#endif
  int const lane = STATIC_TIDX & 63;
  int const e = reinterpret_cast<int const *>(sm + ExpertLds<C>::EID)[slot];
  // the weights and their scales first: they do not wait for h_q, and are in
  // flight while the wait polls (MI355X 10-10: poll, then h_q, then weights
  // was three round trips in a row per item); then h_q, checked by the caller
  // (expert_w2_ready) after the current item's compute and stores
  // tiles t0 .. t0 + W2_TILES - 1
#if STATIC_MOE_W2_HQ_FIRST
  expert_w2_hq<C>(F, sm, slot, hq, hsc);
#endif
  v4i const *wt = reinterpret_cast<v4i const *>(maps.m[C::W2M].p) +
                  ((size_t)e * C::RT2 + t0) * C::KS2 * 64 + lane;
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
      reinterpret_cast<unsigned const *>(maps.m[C::W2SFM].p) +
      (((size_t)e * C::RT2 + t0) / 4 * 64 + lane) * C::SCW;
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
#if !STATIC_MOE_W2_HQ_FIRST
  expert_w2_hq<C>(F, sm, slot, hq, hsc);
#endif
}
// h_q loaded with the item; re-loaded until written
template <class C>
__device__ __forceinline__ void expert_w2_ready(W2Item<C> &F,
                                                char const *sm,
                                                int slot,
                                                unsigned char const *hq,
                                                unsigned char const *hsc) {
  if (expert_w2_ok<C>(F, sm, slot)) {
    return;
  }
#ifdef STATIC_TIMING_BUILD
  long long const tw = gtime();
#endif
  do {
    w2_poll_pause();
    expert_w2_hq<C>(F, sm, slot, hq, hsc);
  } while (!expert_w2_ok<C>(F, sm, slot));
  // a wait the compiler sees: the re-loads are under lane branches, and
  // without it the compiler takes them as maybe pending where this path
  // rejoins, so the compute waited vmcnt(0), for the next item's loads too
  __builtin_amdgcn_s_waitcnt(0);
#ifdef STATIC_TIMING_BUILD
  F.t_wait += gtime() - tw;
#endif
}
struct W2NoHook0 {
  __device__ void operator()() const {}
};
// the item's rows of the slot's tokens: row (t, k) of out = weight x result
template <class C, class HOOK = W2NoHook0>
__device__ __forceinline__ void expert_w2_compute(W2Item<C> &F,
                                                  char const *sm,
                                                  int slot,
                                                  int t0,
                                                  float *out,
                                                  HOOK &&hook = HOOK{}) {
  using Ls = ExpertLds<C>;
  int const lane = STATIC_TIDX & 63, p = lane % 16;
  int const n = reinterpret_cast<int const *>(sm + Ls::NTOK)[slot];
  int t = 0, k = 0;
  float w = 0.f;
  if (p < n) {
    t = reinterpret_cast<int const *>(sm + Ls::TOK)[slot * T + p];
    w = reinterpret_cast<float const *>(sm + Ls::WT)[slot * T + p];
    k = reinterpret_cast<unsigned char const *>(sm + Ls::KK)[slot * T + p];
  }
  float *row_out = out + ((size_t)t * C::K + k) * C::LAT;
#pragma unroll
  for (int i = 0; i < C::W2_TILES; i++) {
    v4f acc = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int kq = 0; kq < C::KS2; kq++) {
      v8i const a = {F.f[i][kq][0], F.f[i][kq][1], F.f[i][kq][2], F.f[i][kq][3], 0, 0, 0, 0};
      acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
          a,
          F.b[kq],
          acc,
          4,
          0,
          0,
          moe_w2_sbyte<C>(F.sc[i / 4], i % 4, kq),
          0,
          F.bs[kq]);
    }
    if (p < n) {
      int const row = (t0 + i) * 16 + 4 * (lane / 16);
      st16_out(row_out + row,
               make_uint4(__float_as_uint(w * acc[0]),
                          __float_as_uint(w * acc[1]),
                          __float_as_uint(w * acc[2]),
                          __float_as_uint(w * acc[3])));
    }
  }
}

// z_q's width from its producer: sum_quant_send's params {L}, or a gemm_tile
// node that quantizes its rows itself (COMBINE_FINISH_Q: params {mode, K, N,
// ...}, z_q's width N)
template <class ZQ>
constexpr int zq_width() {
  if constexpr (sizeof(ZQ::v) / sizeof(int) > 2) {
    return ZQ::v[2];
  } else {
    return ZQ::v[0];
  }
}

// the moe_experts task of one CU (W13CU: a W13 item per CU, its 4 row tiles over the 4 waves)
// RSPLIT > 0 (a local route, T = 1, W13CU): the CU makes the token's K pairs itself from the router's
// RSPLIT K parts (rlog, polled) and the score bias (rbias), with topk_route's code (the same pairs), instead of polling
// topk_route's (MI355X dec62, T = 1: topk_route ended 12.3-13.7 us, the table was built at ~16.5, the W13 weights issued after)
template <class C, bool W13CU = false, int RSPLIT = 0>
__device__ __forceinline__ void moe_experts_task(Maps const &maps,
                                                 G const &g,
                                                 KernelLocals &L,
                                                 unsigned long long const *pairs,
                                                 unsigned char *hq,
                                                 unsigned char *hsc,
                                                 uint32_t *qs,
                                                 unsigned long long *trace,
                                                 float *out,
                                                 float const *rlog = nullptr,
                                                 float const *rbias = nullptr) {
  char *const sm = L.sm;
  __syncthreads(); // the LDS below may still be read by the previous task
#ifdef STATIC_TIMING_BUILD
  long long const t_task0 = gtime(); // the start phase's trace entries (kinds 10, 11)
#endif
  // The items from 16 queues, 8 for the W13 items and 8 for the W2 items, one
  // per die each. Queue x of set q (counter qs[32 (8 q + x)], its own 128-B
  // line) holds the set's items [n x / 8, n (x + 1) / 8); a wave takes from its
  // die's queue, and once that is empty reads all 8 counters of the set (plain
  // loads, one round trip) and takes from one that is not. MI355X 10-10: one
  // queue for all ~1000 waves serialised its adds on one address (~39 us per
  // wave); W2 items dealt statically (wave w: w, w + NW, ...) left the CUs
  // that started late, or had a W13 item, with a full share of W2 items (task
  // median 67 us, longest 93). A wave takes W2 items only after it saw every
  // W13 queue empty: then every W13 item was taken by a wave already in this
  // task, which finishes it without waiting, so a W2 item's wait ends. (Dealing
  // W13 items would not be safe: one dealt to a CU still in an earlier task
  // could hold up a W2 item that task waits for.)
  int const own = xcc_id();
  unsigned dead = 0; // per set q, bit 8 q + x: queue x seen empty
  auto add = [&](int qx) {
    int v = 0;
    if ((STATIC_TIDX & 63) == 0) {
      v = (int)__hip_atomic_fetch_add(
          qs + 32 * qx, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    return __builtin_amdgcn_readfirstlane(v);
  };
  // a ticket: the add on this die's counter issued, its value not read yet (lane 0 holds it; -1: this die's queue of
  // set q was seen empty, no add)
  auto take_issue = [&](int q) -> int {
    if (dead & (1u << (8 * q + own))) {
      return -1;
    }
    int v = 0;
    if ((STATIC_TIDX & 63) == 0) {
      v = (int)__hip_atomic_fetch_add(qs + 32 * own, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    asm volatile("" ::: "memory"); // issued here, before the loads that follow
    return v;
  };
  // One counter per die hands out the die's W13 units, then its W2 items (tickets [0, U) W13 unit, [U, U + N2) W2
  // item): the ticket a wave takes for its next W13 unit names its first W2 item once the die's W13 units are all
  // taken, so it goes on without a round trip (MI355X 10-12, TP8 trace: 120-190 waves at 42-52 us waiting for the W2
  // ticket after their last W13 unit). W13 units from the own die only (MI355X 10-12, TP8 trace: 105-353 of the 1024
  // waves at 44-54 us racing for the other queues' last units, 2.3-6 us each, while ~1000 W2 items were ready); W2
  // items from another die once its W13 units are all taken (read on its counter), so every W2 item's wait ends.
  int w2_pre = -1; // the W2 item the last W13 ticket named (nu2: none left on this die)
  int n13q = 0, n2i = 0; // W13 units, W2 items (set below)
  // the next item of set q (n items) from a ticket, or n when there is none
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
    // W2 items of another die, once its W13 units are all taken
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
  auto take_of = [&](int q, int n) -> int {
    return take_finish(q, n, take_issue(q));
  };
  // the first W13 item's ticket before the waits for
  // z_q and the pairs (its value does not depend on the table, only the item
  // it names does), its round trip overlapping them
  int const tk13 = (W13CU && STATIC_TIDX >= 64) ? -2 : take_issue(0);   // (W13CU: wave 0 takes the CU's W13 items)
  unsigned long long pre_pair = ~0ull;
  if constexpr (RSPLIT > 0) { // (threads 0 .. K - 1 = wave 0's lanes: pair i, as expert_table's pre; scratch in z_q's area)
    static_assert(W13CU && T == 1 && C::NPAIR == C::K &&
                      topk_scratch_bytes<C::NE, C::K>() <= C::KS1 * 64 * 32,
                  "moe_experts: a local route at T = 1 with the table before z_q, topk's scratch in z_q's area");
    pre_pair = topk_route_task<RSPLIT, C::NE, C::K, true>(rbias, const_cast<float *>(rlog), nullptr, 0, sm + ExpertLds<C>::Z,
                                                           false);
  } else if ((int)STATIC_TIDX < C::NPAIR) { // (host-filled with 0xFF before the launch)
    pre_pair = ld64_dev(pairs + STATIC_TIDX);
  }
  // W13CU: the table first (it needs only the pairs, which land before z_q), the
  // first unit's weight loads issued, then z_q (in the W13CU branch below; a CU with no unit does not wait for z_q)
  constexpr bool EARLY = W13CU;
  if constexpr (!EARLY) {
    expert_load_z<C>(g, sm);
    if (STATIC_TIDX == 0) {
      g.stamps[L.sm_id * NSTAMP + STAMP_TASK0] = gtime();
    }
  }
#ifdef STATIC_TIMING_BUILD
  long long const t_zq = gtime(); // (EARLY: the task start, the table follows)
#endif
  int const n_live = expert_table<C>(pairs, sm, pre_pair);
#ifdef STATIC_TIMING_BUILD
  long long const t_tab = gtime();
#endif
#ifdef STATIC_NLIVE_STAMP
  if (L.sm_id == 0 && STATIC_TIDX == 0) { // (diagnosis: the experts used, in CU 0's stamp slot STAMP_ISSUED, unused in normal builds)
    g.stamps[STAMP_ISSUED] = n_live;
  }
#endif
  static_assert(C::LAT / 128 + 1 <= 32, "R_BLOCKS: at most 31 blocks");
  int const n13 = n_live * C::ITEMS1, n_all = n13 + n_live * C::ITEMS2;
  int const nu = n13; // W13 units: one per item (queue x: the items [n13 x / 8, n13 (x + 1) / 8))
  n13q = nu;
  n2i = n_all - n13;
#ifdef STATIC_TIMING_BUILD
  // the timing build: per CU, its waves' W13 and W2 item counts and W13 item
  // time (stage stamp 0, packed) and when its last wave left the W13 items
  // (stage stamp 1, globaltimer ns), for amd/tests/test_layer_one_gpu.py
  __shared__ unsigned long long t_take_s, t_w13_s, t_w2x_s;
  if (STATIC_TIDX == 0) {
    t_take_s = 0;
    t_w13_s = 0;
    t_w2x_s = 0;
  }
  __syncthreads();
  long long t_w13 = 0, t_take2 = 0, t_w2wait = 0;
  unsigned n_w13 = 0, n_w2 = 0; // this wave's items
  // the item trace: this wave's MOE_TRACE_ITEMS entries of 2 u64: start ns,
  // duration ns | kind (1 W13, 2 W2) << 32 | slot << 40
  constexpr int MOE_TRACE_ITEMS = 32;
  unsigned long long *const tr =
      trace + ((size_t)L.sm_id * 4 + (STATIC_TIDX >> 6)) * MOE_TRACE_ITEMS * 2;
  int n_tr = 0;
  auto trace_item = [&](long long t0, long long t1, int kind, int slot) {
    if ((STATIC_TIDX & 63) == 0 && n_tr < MOE_TRACE_ITEMS) {
      tr[2 * n_tr] = (unsigned long long)t0;
      tr[2 * n_tr + 1] = (unsigned long long)(t1 - t0) |
                         ((unsigned long long)kind << 32) |
                         ((unsigned long long)slot << 40);
    }
    n_tr++;
  };
  // the shader clock over the wave's items: two trace entries at the end, kind
  // 8 (real time, ns) and kind 9 (s_memtime cycles in the duration field), so
  // the clock (cycles / ns) of solo and TP8 runs can be compared (MI355X 10-12:
  // W13 / W2 units 15-40% longer in TP8 than solo, the front's GEMMs not)
  long long const clk_t0 = gtime();
  long long const clk_c0 = (long long)__builtin_amdgcn_s_memtime();
  // the start phase: kind 10 task start -> z_q in LDS, kind 11 -> the routing table built
  trace_item(t_task0, t_zq, 10, 0);
  trace_item(t_zq, t_tab, 11, 0);
#else
  (void)trace;
#endif
  // W13 items: taken one at a time (MI355X 10-11, solo: a ticket taken ahead, as
  // the W2 items below, let a wave hold two items while CUs that start later
  // found the queue empty: e2e +5-10 us; an item per wave pair, gate / up tiles
  // split, 2 x the K steps in flight: no change; each item as two K-part units,
  // 16 + 12 steps, partial sums combined by the second: span +3-6 us). The next
  // unit's ticket is issued inside the current unit, after its last weight loads
  // (expert_w13_acc's hook), its round trip hidden behind the unit's own (MI355X
  // 10-12, TP8 trace: a wave's last W13 unit end -> its first W2 load 3.6-4.4 us
  // before, three round trips in a row). Held for the rest of the unit: a few
  // us, not a whole unit (a ticket taken at the unit's start held items from the
  // CUs that join late, 10-11).
  int tkn = -2; // -2: none issued
  auto tk_hook1 = [&]() { tkn = take_issue(0); };
  auto next13 = [&]() -> int {
    int const t = tkn;
    tkn = -2;
    return t != -2 ? take_finish(0, nu, t) : take_of(0, nu);
  };
  auto w13_unit = [&](int j) {
#ifdef STATIC_TIMING_BUILD
    long long const t13 = gtime();
    long long t_sum = 0; // the unit's sums made (its loads all landed): trace kind 13
#endif
    int const slot = j / C::ITEMS1, it = j % C::ITEMS1;
    {
      v4f acc[4]; // (expert_w13, with the ticket hook)
      expert_w13_acc<C, 0, 4, C::STAGES>(maps, sm, slot, it, acc, tk_hook1);
#ifdef STATIC_TIMING_BUILD
      t_sum = gtime();
#endif
      expert_w13_finish<C>(sm, slot, it, acc, acc + 2, hq, hsc);
    }
#ifdef STATIC_TIMING_BUILD
    long long const t13e = gtime();
    t_w13 += t13e - t13;
    n_w13++;
    trace_item(t13, t13e, 1, slot);
    trace_item(t_sum, t13e, 13, slot); // sums made -> unit end (SiTU, h_q)
#endif
  };
  if constexpr (W13CU) {
    // the CU's items: wave 0 takes one (its ticket for the next one issued once it is read), every wave issues its tile's
    // loads, computes, the sums meet in LDS (XCH), wave 0 makes h_q
    int *const cuq = reinterpret_cast<int *>(sm + ExpertLds<C>::CUQ);
    v4f *const xch = reinterpret_cast<v4f *>(sm + ExpertLds<C>::XCH);
    int const wv = STATIC_TIDX >> 6, ln = STATIC_TIDX & 63;
    int tkc = tk13;
    bool zq = !EARLY; // z_q in LDS
    if constexpr (STATIC_MOE_ILV > 0) {
      constexpr int E = STATIC_MOE_ILV, Q2 = (C::ITEMS2 + 3) / 4;
      int const e_lo = n_live * own / 8, nexp = n_live * (own + 1) / 8 - e_lo, G = (nexp + E - 1) / E;
      int const M = nexp * (C::ITEMS1 + Q2);
      uint32_t *const rblk_ = qs + MOE_MISC_RBLK / 4;
      int prev_t0 = -1; // the last W2 item's R tiles, counted after a later load wait (as w2_lflush_old)
      auto post_prev = [&]() {
        if (prev_t0 >= 0 && ln == 0) {
          __hip_atomic_fetch_add(rblk_ + RBLK_STRIDE * (1 + prev_t0 / 8), (uint32_t)C::W2_TILES, __ATOMIC_RELAXED,
                                 __HIP_MEMORY_SCOPE_AGENT);
        }
        prev_t0 = -1;
      };
      int tkv = -2; // wave 0: the ticket issued for the next entry
#pragma unroll 1
      for (int first = 1;; first = 0) {
        __syncthreads(); // every wave read cuq / xch of the last entry
        if (wv == 0) {
          int const v = __builtin_amdgcn_readfirstlane(first ? tkc : tkv);
          if (ln == 0) {
            *cuq = v;
          }
          if (v < M) {
            tkv = take_issue(0);
          }
        }
        __syncthreads();
        int const v = *cuq;
        if (v >= M) {
          break;
        }
        // the entry: walk the blocks W13 (0), W13 (1), W2 (0), W13 (2), W2 (1), .., W2 (G - 1)
        int rem = v, kind = 0, grp = 0;
#pragma unroll 1
        for (int k = 0;; k++) {
          int kd, gg;
          if (k == 0) {
            kd = 0; gg = 0;
          } else if (k == 1 && G > 1) {
            kd = 0; gg = 1;
          } else {
            int const m = k - (G > 1 ? 2 : 1); // 0: W2 (0), 1: W13 (2), 2: W2 (1), ..
            if ((m & 1) == 0 || m / 2 + 2 >= G) {
              kd = 1; gg = (m & 1) == 0 ? m / 2 : m / 2 + 1;
            } else {
              kd = 0; gg = m / 2 + 2;
            }
          }
          int const ne = (nexp - gg * E) < E ? nexp - gg * E : E;
          int const sz = ne * (kd == 0 ? C::ITEMS1 : Q2);
          if (rem < sz) {
            kind = kd;
            grp = gg;
            break;
          }
          rem -= sz;
        }
        if (kind == 0) { // a W13 unit: the CU's 4 waves
          int const slot = e_lo + grp * E + rem / C::ITEMS1, jc = slot * C::ITEMS1 + rem % C::ITEMS1;
#ifdef STATIC_TIMING_BUILD
          long long const t13 = gtime();
#endif
          W13Tile<C> F;
          w13_tile_issue<C>(F, maps, sm, slot, jc % C::ITEMS1, wv);
          if (!zq) {
            expert_load_z<C>(g, sm);
            __syncthreads();
            if (STATIC_TIDX == 0) {
              g.stamps[L.sm_id * NSTAMP + STAMP_TASK0] = gtime();
            }
            zq = true;
          }
          xch[wv * 64 + ln] = w13_tile_compute<C>(F, sm);
          __syncthreads();
          if (wv == 0) {
            v4f const gate[2] = {xch[ln], xch[64 + ln]}, up[2] = {xch[128 + ln], xch[192 + ln]};
            expert_w13_finish<C>(sm, slot, jc % C::ITEMS1, gate, up, hq, hsc);
          }
#ifdef STATIC_TIMING_BUILD
          long long const t13e = gtime();
          t_w13 += t13e - t13;
          n_w13++;
          trace_item(t13, t13e, 1, slot);
#endif
        } else { // 4 W2 items of the expert, one per wave
          int const slot = e_lo + grp * E + rem / Q2, it2 = 4 * (rem % Q2) + wv;
          if (it2 < C::ITEMS2) {
            int const u = slot * C::ITEMS2 + it2, t0 = it2 * C::W2_TILES;
            W2Item<C> A;
            expert_w2_load<C>(A, maps, g, sm, slot, t0, hq, hsc);
            expert_w2_ready<C>(A, sm, slot, hq, hsc);
            post_prev(); // (a load wait since its stores)
            expert_w2_compute<C>(A, sm, slot, t0, out);
            prev_t0 = t0;
            (void)u;
#ifdef STATIC_TIMING_BUILD
            trace_item(A.t_load, gtime(), 2, slot);
            n_w2++;
#endif
          }
        }
      }
      wait_mem();
      post_prev();
    } else {
#pragma unroll 1
    for (;;) {
      __syncthreads(); // every wave read cuq / xch of the last item
      if (wv == 0) {
        int const v = take_finish(0, nu, tkc);
        if (ln == 0) {
          *cuq = v;
        }
      }
      __syncthreads();
      int const jc = *cuq;
      if (jc >= nu) {
        break;
      }
      if (wv == 0) {
        tkc = take_issue(0);
      }
#ifdef STATIC_TIMING_BUILD
      long long const t13 = gtime();
#endif
      W13Tile<C> F;
      w13_tile_issue<C>(F, maps, sm, jc / C::ITEMS1, jc % C::ITEMS1, wv);
      if (!zq) { // (EARLY, the first unit: its weights in flight while z_q lands)
        expert_load_z<C>(g, sm);
        __syncthreads();
        if (STATIC_TIDX == 0) {
          g.stamps[L.sm_id * NSTAMP + STAMP_TASK0] = gtime();
        }
        zq = true;
      }
      xch[wv * 64 + ln] = w13_tile_compute<C>(F, sm);
      __syncthreads();
      if (wv == 0) {
        v4f const gate[2] = {xch[ln], xch[64 + ln]}, up[2] = {xch[128 + ln], xch[192 + ln]};
        expert_w13_finish<C>(sm, jc / C::ITEMS1, jc % C::ITEMS1, gate, up, hq, hsc);
      }
#ifdef STATIC_TIMING_BUILD
      long long const t13e = gtime();
      t_w13 += t13e - t13;
      n_w13++;
      trace_item(t13, t13e, 1, jc / C::ITEMS1);
#endif
    }
    }
    if (!zq && STATIC_TIDX == 0) {
      g.stamps[L.sm_id * NSTAMP + STAMP_TASK0] = gtime();
    }
  } else if constexpr (STATIC_MOE_ILVW > 0) {
    constexpr int E = STATIC_MOE_ILVW, I13 = C::ITEMS1; // (W13 entries per expert)
    int const e_lo = n_live * own / 8, nexp = n_live * (own + 1) / 8 - e_lo, G = (nexp + E - 1) / E;
    int const M = nexp * (I13 + C::ITEMS2);
    uint32_t *const rblk_ = qs + MOE_MISC_RBLK / 4;
    int const ln_ = STATIC_TIDX & 63;
    int prev_t0 = -1;
    auto post_prev = [&]() {
      if (prev_t0 >= 0 && ln_ == 0) {
        __hip_atomic_fetch_add(rblk_ + RBLK_STRIDE * (1 + prev_t0 / 8), (uint32_t)C::W2_TILES, __ATOMIC_RELAXED,
                               __HIP_MEMORY_SCOPE_AGENT);
      }
      prev_t0 = -1;
    };
    int tk = tk13;
#pragma unroll 1
    for (;;) {
      int const v = __builtin_amdgcn_readfirstlane(tk);
      if (v >= M) {
        break;
      }
      int rem = v, kind = 0, grp = 0;
#pragma unroll 1
      for (int k = 0;; k++) { // the blocks W13 (0), W13 (1), W2 (0), W13 (2), W2 (1), .., W2 (G - 1)
        int kd, gg;
        if (k == 0) {
          kd = 0; gg = 0;
        } else if (k == 1 && G > 1) {
          kd = 0; gg = 1;
        } else {
          int const m = k - (G > 1 ? 2 : 1);
          if ((m & 1) == 0 || m / 2 + 2 >= G) {
            kd = 1; gg = (m & 1) == 0 ? m / 2 : m / 2 + 1;
          } else {
            kd = 0; gg = m / 2 + 2;
          }
        }
        int const ne = (nexp - gg * E) < E ? nexp - gg * E : E;
        int const sz = ne * (kd == 0 ? I13 : C::ITEMS2);
        if (rem < sz) {
          kind = kd;
          grp = gg;
          break;
        }
        rem -= sz;
      }
      if (kind == 0) { // a W13 item (the next ticket issued after its last weight loads)
        int const slot = e_lo + grp * E + rem / I13, it = rem % I13;
#ifdef STATIC_TIMING_BUILD
        long long const t13 = gtime();
#endif
        auto hook = [&]() { tk = take_issue(0); };
      v4f acc[4];
      expert_w13_acc<C, 0, 4, C::STAGES>(maps, sm, slot, it, acc, hook);
      expert_w13_finish<C>(sm, slot, it, acc, acc + 2, hq, hsc);
      
#ifdef STATIC_TIMING_BUILD
        long long const t13e = gtime();
        t_w13 += t13e - t13;
        n_w13++;
        trace_item(t13, t13e, 1, slot);
#endif
      } else { // a W2 item
        int const slot = e_lo + grp * E + rem / C::ITEMS2, t0 = (rem % C::ITEMS2) * C::W2_TILES;
        tk = take_issue(0);
        W2Item<C> A;
        expert_w2_load<C>(A, maps, g, sm, slot, t0, hq, hsc);
        expert_w2_ready<C>(A, sm, slot, hq, hsc);
        post_prev(); // (a load wait since its stores)
        expert_w2_compute<C>(A, sm, slot, t0, out);
        prev_t0 = t0;
#ifdef STATIC_TIMING_BUILD
        trace_item(A.t_load, gtime(), 2, slot);
        n_w2++;
#endif
      }
    }
    wait_mem();
    post_prev();
  } else {
  int j = take_finish(0, nu, tk13);
#ifdef STATIC_EXPERT_STAMPS
  if (STATIC_TIDX == 0 && j < nu) {
    g.stamps[L.sm_id * NSTAMP + 4] = gtime();
  }
#endif
#pragma unroll 1
  while (j < nu) { // W13 units
    w13_unit(j);
    j = next13();
  }
  }
#ifdef STATIC_EXPERT_STAMPS
  if (STATIC_TIDX == 0) {
    g.stamps[L.sm_id * NSTAMP + 5] = gtime();
  }
#endif
#ifdef STATIC_TIMING_BUILD
  long long const t_w13end = gtime();
#endif
  // W2 items (of n2) from their queues: two in flight, A and B alternating
  int const n2 = n_all - n13, nu2 = n2;

#ifdef STATIC_TIMING_BUILD
  long long tt = gtime();
#endif
  int ia = ((W13CU && STATIC_MOE_ILV > 0) || (!W13CU && STATIC_MOE_ILVW > 0)) ? nu2
                                                                               : ((w2_pre >= 0 && w2_pre < nu2) ? w2_pre : take_of(1, nu2));
#ifdef STATIC_TIMING_BUILD
  t_take2 += gtime() - tt;
  n_w2 += ia < nu2;
#endif
  // the next item's ticket is issued BEFORE the current item's loads and read
  // after them: vmcnt counts in issue order, so reading it then waits for the
  // ticket alone, not for the loads issued after it (MI355X 10-11, solo: the W2
  // takes 71 -> 18 us per CU over its 4 waves, e2e 121 -> 109 us; taken after
  // the loads, the add's value waited behind the previous item's weights)
  // R in blocks: the blocks' counters (core.cuh MOE_MISC_RBLK); counter 0
  // n_live, stored by CU 0 only (every CU has the same table) and after its W13
  // items (MI355X 10-11, solo: every CU's store to that word right after the
  // table, before the W13 items: the last W13 end 51-57 -> 60-65 us; each CU's
  // first wait for W13 loads also waited for its store, in line behind the
  // other CUs' stores to the same address)
  uint32_t *const rblk = qs + MOE_MISC_RBLK / 4;
  if (STATIC_TIDX == 0 && L.sm_id == 0) {
    __hip_atomic_store(rblk, (uint32_t)n_live, __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
  }
  // a unit's R tiles are counted per 128-column block once its stores are
  // complete: after a later load wait (which drains every earlier memory
  // operation), not right after the stores (MI355X 10-11, solo: a wait for the
  // stores after every unit, e2e +6 us). The counts trail two units (a unit's
  // stores are complete once a load issued after them is).
  int pend_t0 = -1, pend_nt = 0;
  auto w2_rflush = [&]() {
    if (pend_t0 >= 0) {
      wait_mem(); // free after a load wait; at the end the one wait
      if ((STATIC_TIDX & 63) == 0) {
        for (int b = pend_t0 / 8; b <= (pend_t0 + pend_nt - 1) / 8; b++) {
          int const lo = pend_t0 > 8 * b ? pend_t0 : 8 * b,
                    hi = pend_t0 + pend_nt < 8 * b + 8 ? pend_t0 + pend_nt
                                                       : 8 * b + 8;
          __hip_atomic_fetch_add(rblk + RBLK_STRIDE * (1 + b), (uint32_t)(hi - lo),
                                 __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
      }
      pend_t0 = -1;
    }
  };
  int pold_t0 = -1, pold_nt = 0;
  auto w2_lpend = [&](int t0_, int nt_) {
    if (pold_t0 >= 0) { // two pending (only at the end): the older now
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
  auto w2_lflush_old = [&]() { // after a wait for loads issued after its stores
    if (pold_t0 >= 0 && (STATIC_TIDX & 63) == 0) {
      for (int b = pold_t0 / 8; b <= (pold_t0 + pold_nt - 1) / 8; b++) {
        int const lo = pold_t0 > 8 * b ? pold_t0 : 8 * b,
                  hi = pold_t0 + pold_nt < 8 * b + 8 ? pold_t0 + pold_nt
                                                     : 8 * b + 8;
        __hip_atomic_fetch_add(rblk + RBLK_STRIDE * (1 + b), (uint32_t)(hi - lo),
                               __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
    pold_t0 = -1;
  };
  // a wave's W2 step: issue the next item's loads, compute and store the current item, then check the next item's h_q
  auto w2_load = [&](W2Item<C> &F, int u) {
    expert_w2_load<C>(F, maps, g, sm, u / C::ITEMS2, (u % C::ITEMS2) * C::W2_TILES, hq, hsc);
  };
  auto w2_ready = [&](W2Item<C> &F, int u) { expert_w2_ready<C>(F, sm, u / C::ITEMS2, hq, hsc); };
  auto w2_compute = [&](W2Item<C> &F, int u, auto &&hook) {
    int const t0 = (u % C::ITEMS2) * C::W2_TILES;
    expert_w2_compute<C>(F, sm, u / C::ITEMS2, t0, out, hook);
    w2_lpend(t0, C::W2_TILES);
#ifdef STATIC_TIMING_BUILD
    trace_item(F.t_load, gtime(), 2, u / C::ITEMS2);
    n_w2++;
#endif
  };
  if (W13CU && STATIC_W2_ONE && ia < nu2) {
#pragma unroll 1
    for (;;) {
      W2Item<C> A;
      w2_load(A, ia);
      w2_ready(A, ia);
      w2_lflush_old();
      int const tk = take_issue(1); // (claimed once this item's h_q is here; its round trip during the compute)
      w2_compute(A, ia, W2NoHook0{});
      ia = take_finish(1, nu2, tk);
      if (ia >= nu2) {
        break;
      }
    }
  } else if (ia < nu2) {
    W2Item<C> A, B;
#ifdef STATIC_EXPERT_STAMPS
    if (STATIC_TIDX == 0) {
      g.stamps[L.sm_id * NSTAMP + 6] = gtime();
    }
#endif
    int tk = take_issue(1);
    w2_load(A, ia);
    w2_ready(A, ia);
#pragma unroll 1
    for (;;) {
      int const ib = take_finish(1, nu2, tk);
      if (ib >= nu2) {
        w2_compute(A, ia, W2NoHook0{});
        break;
      }
      tk = take_issue(1);
      w2_load(B, ib);
      w2_compute(A, ia, W2NoHook0{});
      w2_ready(B, ib);
      w2_lflush_old();
      ia = take_finish(1, nu2, tk);
      if (ia >= nu2) {
        w2_compute(B, ib, W2NoHook0{});
        break;
      }
      tk = take_issue(1);
      w2_load(A, ia);
      w2_compute(B, ib, W2NoHook0{});
      w2_ready(A, ia);
      w2_lflush_old();
    }
  }
#ifdef STATIC_EXPERT_STAMPS
  if (STATIC_TIDX == 0) {
    g.stamps[L.sm_id * NSTAMP + 7] = gtime();
  }
#endif
  // the last ones: wait for their stores, then count
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
  if ((STATIC_TIDX & 63) == 0) {
    // stage stamp 0: W13 items | W2 items << 16 | W13 item time (us, all waves) << 32
    atomicAdd(&t_take_s, (unsigned long long)n_w13 | ((unsigned long long)n_w2 << 16) |
                             ((unsigned long long)(t_w13 / 1000) << 32));
    atomicMax(&t_w13_s, (unsigned long long)t_w13end);
    atomicAdd(&t_w2x_s, (unsigned long long)(t_w2wait / 1000) | ((unsigned long long)(t_take2 / 1000) << 10));
  }
  __syncthreads();
  if (STATIC_TIDX == 0 && L.stage_stamps) {
    L.stage_stamps[0] = t_take_s;
    // stage stamp 1: the last wave's W13 end (ns, 44 bits) | W2 h_q wait (us, all waves, 10 bits) << 44 | W2 take (us, all
    // waves, 10 bits) << 54
    L.stage_stamps[1] = (t_w13_s & ((1ull << 44) - 1)) | ((t_w2x_s & 0x3FFu) << 44) | (((t_w2x_s >> 10) & 0x3FFu) << 54);
  }
#endif
  // allreduce_send waits for R's blocks, not for a count of the CUs done (a wait
  // for the CU's stores, a barrier and an add on one counter that 256 CUs share,
  // at the end of the last CUs' moe_experts)
  __syncthreads();
}

} // namespace static_mk
#include "moe_experts_csr.cuh" // T > 16: the table in CSR form, z_q staged per W13 unit, h_q by pair
namespace static_mk {

// the CSR path (moe_experts_csr.cuh): T > 16, or params[2] = 1 (static_megakernel.moe_experts_layer: MPK_MOE_CSR=1, any T)
template <class PARAMS>
constexpr bool moe_csr() {
  if constexpr (T > 16) {
    return true;
  } else if constexpr (sizeof(PARAMS::v) / sizeof(int) > 3) {
    return PARAMS::v[2] == 1;
  } else {
    return false;
  }
}

template <class A, class... B>
struct moe_first_input {
  using type = A;
};

// moe_experts task (one per CU): ZQ = sum_quant_send (z_q, its params[0] =
// LAT), PAIRS = topk_route (its params {NE, K}); PARAMS {IR}; BUF_SLOTS {HQ,
// HSF, the work queues ([16][32] u32, 0 before each launch)}; MAP_SLOTS as above
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class ZQ,
          class PAIRS,
          class... REST>
__device__ __forceinline__ void run_moe_experts(Maps const &maps,
                                                G const &g,
                                                KernelLocals &L,
                                                StaticTask const &) {
  static_assert(
      SELF::buf >= 0 && SELF::counter >= 0 && PAIRS::buf >= 0,
      "moe_experts: its output rows, its counter, topk_route's pairs");
  static_assert(ZQ::buf >= 0 && exchange_bytes[ZQ::buf] == zq_bytes(zq_width<ZQ>()),
                "moe_experts: z_q in the exchange region");
  if constexpr (moe_csr<PARAMS>()) {
    using CC = ExpertCfgC<PAIRS::v[0], PAIRS::v[1], zq_width<ZQ>(), PARAMS::v[0], SELF::counter, exchange_offset[ZQ::buf],
                          slot_at<MAP_SLOTS, 0>(), slot_at<MAP_SLOTS, 2>(), slot_at<MAP_SLOTS, 4>(), slot_at<MAP_SLOTS, 6>()>;
    moe_experts_csr_task<CC>(maps, g, L, buf_at<unsigned long long const>(g, PAIRS::buf),
                             buf_at<unsigned char>(g, slot_at<BUF_SLOTS, 0>()), buf_at<unsigned char>(g, slot_at<BUF_SLOTS, 1>()),
                             buf_at<uint32_t>(g, slot_at<BUF_SLOTS, 2>()),
                             reinterpret_cast<unsigned long long *>(buf_at<unsigned char>(g, slot_at<BUF_SLOTS, 2>()) +
                                                                    CC::MISC_TRACE),
                             buf_at<float>(g, SELF::buf));
  } else {
  // params[1] 1: a W13 item per CU (moe_experts_task W13CU; few items: T <= 4)
  constexpr bool W13CU = []() constexpr {
    if constexpr (sizeof(PARAMS::v) / sizeof(int) > 2) {
      return PARAMS::v[1] == 1;
    } else {
      return false;
    }
  }();
  using C = ExpertCfg<PAIRS::v[0],
                      PAIRS::v[1],
                      zq_width<ZQ>(),
                      PARAMS::v[0],
                      SELF::counter,
                      exchange_offset[ZQ::buf],
                      slot_at<MAP_SLOTS, 0>(),
                      slot_at<MAP_SLOTS, 2>(),
                      slot_at<MAP_SLOTS, 4>(),
                      slot_at<MAP_SLOTS, 6>()>;
  // params[3] 1: a local route (the third input the router's node, buffer slot 4 the score bias)
  constexpr bool LOCAL = []() constexpr {
    if constexpr (sizeof(PARAMS::v) / sizeof(int) > 3) {
      return PARAMS::v[3] == 1;
    } else {
      return false;
    }
  }();
  if constexpr (LOCAL) {
    using LOG = typename moe_first_input<REST...>::type;
    static_assert(LOG::v[0] == COMBINE_SLOTS && LOG::v[2] == PAIRS::v[0] && LOG::buf >= 0,
                  "moe_experts: a local route reads the router's NE logits, its K parts in slots");
    moe_experts_task<C, W13CU, LOG::y>(maps, g, L, buf_at<unsigned long long const>(g, PAIRS::buf),
                                       buf_at<unsigned char>(g, slot_at<BUF_SLOTS, 0>()),
                                       buf_at<unsigned char>(g, slot_at<BUF_SLOTS, 1>()),
                                       buf_at<uint32_t>(g, slot_at<BUF_SLOTS, 2>()),
                                       reinterpret_cast<unsigned long long *>(
                                           buf_at<unsigned char>(g, slot_at<BUF_SLOTS, 2>()) + C::MISC_TRACE),
                                       buf_at<float>(g, SELF::buf), buf_at<float const>(g, LOG::buf),
                                       buf_at<float const>(g, slot_at<BUF_SLOTS, 4>()));
  } else {
  moe_experts_task<C, W13CU>(maps,
                      g,
                      L,
                      buf_at<unsigned long long const>(g, PAIRS::buf),
                      buf_at<unsigned char>(g, slot_at<BUF_SLOTS, 0>()),
                      buf_at<unsigned char>(g, slot_at<BUF_SLOTS, 1>()),
                      // slot 2: the queues, then the trace
                      buf_at<uint32_t>(g, slot_at<BUF_SLOTS, 2>()),
                      reinterpret_cast<unsigned long long *>(
                          buf_at<unsigned char>(g, slot_at<BUF_SLOTS, 2>()) +
                          C::MISC_TRACE),
                      buf_at<float>(g, SELF::buf));
  }
  }
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class ZQ,
          class PAIRS,
          class... REST>
constexpr int smem_moe_experts() {
  if constexpr (moe_csr<PARAMS>()) {
    return ExpertLdsC<ExpertCfgC<PAIRS::v[0], PAIRS::v[1], zq_width<ZQ>(), PARAMS::v[0], SELF::counter, 0>>::BYTES;
  } else {
  return ExpertLds<ExpertCfg<PAIRS::v[0],
                             PAIRS::v[1],
                             zq_width<ZQ>(),
                             PARAMS::v[0],
                             SELF::counter,
                             0>>::BYTES;
  }
}

} // namespace static_mk
