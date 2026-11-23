// tasks/swiglu_moe_experts.cuh -- the routed experts of DeepSeek-V4.1-Flash
// (inference/model.py Expert.forward, MoE.forward's loop), after routing:
//   weights  MXFP4: e2m1 with an e8m0 scale per 32 (the checkpoint's, unchanged)
//   x (x_q)  e4m3 with an e8m0 scale per 32: act_quant's output (a node buffer)
//   h (h_q)  the same format, act_quant's formula (amax floor 1e-4)
//   act      gate and up rounded to bf16 (fp4_gemm's output), up clamped to
//            [-L, L] and gate to <= L, h = bf16(silu(gate) * up); the routing
//            weight multiplies the expert's W2 output (SGLang's and vLLM's
//            order: their trtllm-gen MoE applies it in the finalize after W2;
//            the official model.py applies it to h, before the bf16 rounding
//            and act_quant for W2)
// This task follows moe_experts.cuh (K3: MXFP4 x MXFP8 too, SiTU) and reuses
// its routing table (ExpertSlots, table_poll, build_table), its queue
// (QueueEntry and the order below), its warp roles and its W13 ring jobs
// (runtime.cuh load_job, issue_job_warp). What differs:
//   1. x_q is act_quant's node buffer, waited on through its counter (K3: z_q
//      in the exchange region, two sets).
//   2. W13 epilogue (an item = expert slot, 64 features: 64 gate rows + 64 up
//      rows of the expert x x_q, K = LAT): h as above -> h_q and its scale
//      chunk, then CHQ[slot] += 1.
//   3. W2 runs through the same ring as W13 (K3: a ring of whole W2 tiles over
//      the W13 ring's area, which does not fit K = 640: a 5-K-tile stage is 92
//      KB). A W2 tile (slot, 128-row output tile) is a ring job of (KT2 + 1) /
//      2 stages; each stage carries up to 2 weight pieces and their scale
//      chunks, the slot's h_q K tiles and their scale chunks (OFF_HSF); its
//      issuer copies the h_q scales to TMEM 272 + 4 (2 st + at).
//   4. W2 epilogue: row (t, k) of R = weight(t, expert) x the result, plain
//      store (as moe_experts.cuh).
// The intermediate on this GPU is I (the node's params[0]; TP4: 576 = 9 W13
// items); W2's K is I padded to IRP = whole K tiles of 128 (640): h_q's columns
// [I, IRP) are never written (zeroed scratch) and W2's weight columns there are
// zero. The queue: all W13 items, then W2 entries of 2 tiles, the last NSM
// tiles in 1-tile entries (K3's order). R's row (t, k): the pairs are sorted by
// expert id (topk_route, the DeepSeek-V4 gate), so k counts the token's experts
// in ascending id. TMEM columns: 0..63 accumulators, 72 + 4 kt x_q's KT_LAT
// scale chunks (phase switch), 232 + 4 (2 st + at) the weight scales of ring
// stage st, piece at, 272 + 4 (2 st + at) a W2 stage's h_q scales. T > 8
// (11-21): the accumulators take 2 ACC_STAGES T columns (T = 16: 0..127, 32:
// 0..191), x_q's scale chunks move past the h_q scales (TM_ZQ); the epilogues
// drain T columns.
#pragma once
#include "../core.cuh"
#include "act_quant.cuh"   // x_q's layout, xq_exponent, bf16r
#include "moe_experts.cuh" // ExpertSlots, QueueEntry, table_poll, build_table

namespace static_mk {
namespace swiglu_moe {

// what the bodies below use of the layer state (run_swiglu_moe_experts fills it
// from the node's slots): ExpertSlots' (its ZQ / ZQ_KT / ZQ_SF are x_q's maps,
// HQ_KT h_q's 2-K-tile map, HQ_SF its 1-chunk scale map; IR = IRP), plus HQ_A8
// (h_q, one K tile) and HQ_SF2 (its scales, 2 chunks), I, the limit L (fp32
// bits), MT13 = I / 64 W13 items per expert
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
          int ZQ_KT_,
          int ZQ_SF_,
          int HQ_KT_,
          int HQ_SF_,
          int HQ_A8_,
          int HQ_SF2_,
          int NE_,
          int K_,
          int LAT_,
          int I_,
          int LIMIT_BITS_>
struct Slots : ExpertSlots<HQ_,
                           HSF_,
                           CNT_,
                           W13_,
                           W13X2_,
                           W13SF_,
                           W13SFX2_,
                           W2_,
                           W2X2_,
                           W2SF_,
                           W2SFX2_,
                           ZQ_,
                           ZQ_KT_,
                           ZQ_SF_,
                           HQ_KT_,
                           HQ_SF_,
                           0,
                           NE_,
                           K_,
                           LAT_,
                           (I_ + 127) / 128 * 128> {
  static constexpr int HQ_A8 = HQ_A8_, HQ_SF2 = HQ_SF2_, I = I_,
                       IRP = (I_ + 127) / 128 * 128, LIMIT_BITS = LIMIT_BITS_;
  static constexpr int MT13 = I_ / 64;         // W13 items per expert
  static constexpr int NST2 = (IRP / 128 + 1) / 2; // ring stages of a W2 tile
  static_assert(I_ % 64 == 0 && NST2 >= 2,
                "swiglu_moe_experts: whole 64-feature W13 items; a W2 tile of "
                "at least 2 stages (both issuers get one)");
};

// ---- shared memory after the ring (offsets from the dynamic base) ----
// TMEM columns: a W2 stage's h_q scales from TM_HQ (272; T = 64: past the
// weight scales at TM_W13_SF = 256), x_q's first scale chunk at TM_ZQ (T = 8:
// 72, between the accumulators and the weight scales; T > 8: past the h_q
// scales)
constexpr int TM_HQ = T == 64 ? TM_W13_SF + 4 * 2 * SMAX : 272;
constexpr int TM_ZQ = T == 8 ? 72 : TM_HQ + 4 * 2 * SMAX;

template <class RES>
struct Smem {
  static constexpr int OFF_ZQ_SF = OFF_W,          // x_q's scale chunks, staged
                                                    // once at the phase switch
      OFF_HSF = RING_BYTES,                        // SMAX x 2 h_q scale chunks
      OFF_ACC = OFF_HSF + SMAX * 2 * SF_CHUNK;     // a W13 item's accumulator
  static constexpr int BYTES = OFF_ACC + 128 * T * 4;
  static_assert(RES::KT_LAT * SF_CHUNK <= OFF_WSF,
                "the x_q scale chunks in the ring area");
  static_assert(T == 8 ? 72 + 4 * RES::KT_LAT <= 232 && 272 + 4 * 2 * SMAX <= 512
                       : 2 * ACC_STAGES * T <= TM_W13_SF &&
                             TM_W13_SF + 4 * 2 * SMAX <= TM_HQ &&
                             TM_ZQ + 4 * RES::KT_LAT <= 512,
                "swiglu_moe_experts: the TMEM columns of the scales");
};

// loader: arm the stages of W2 tile (slot, ot) with s % 2 == par (two loaders
// split the stages, as load_job): wait for the ring slot, load the weight
// pieces and their scale chunks, and once all W13 items of the slot are done
// (CHQ counter, acquire, then the async-proxy fence) its h_q K tiles and their
// scale chunks. g: the ring stage counter (advanced by every stage); chk: the
// slot this loader has checked last
template <class RES>
__device__ __forceinline__ void load_w2_job(Maps const &maps,
                                            G const &g,
                                            uint32_t base,
                                            int &gq,
                                            int par,
                                            int slot,
                                            int ex,
                                            int ot,
                                            int &chk) {
  constexpr int KT2 = RES::KT2, NST = RES::NST2;
  int st = gq % SMAX;
  uint32_t ph = ((gq / SMAX) - 1) & 1;
  int const p0 = ex * (RES::OT2 * KT2) + ot * KT2;
  for (int s = 0; s < NST; s++, gq++) {
    if ((s & 1) == par) {
      if (gq >= SMAX) {
        mbar_wait(rt.empty0 + 8 * st, ph);
      }
      int const na = (KT2 - 2 * s < 2) ? KT2 - 2 * s : 2;
      uint32_t const full = rt.full0 + 8 * st, dst = base + st * FSTAGE,
                     sf_dst = base + OFF_WSF + st * 2 * SF_CHUNK,
                     hsf_dst = base + Smem<RES>::OFF_HSF + st * 2 * SF_CHUNK;
      mbar_expect(full, na * (8192 + SF_CHUNK + T * 128 + SF_CHUNK));
      if (na == 2) {
        tma3(&maps.m[RES::W2X2], full, dst, 0, 0, p0 + 2 * s, EVICT_FIRST);
        tma3(&maps.m[RES::W2SFX2], full, sf_dst, 0, 0, p0 + 2 * s, EVICT_FIRST);
      } else {
        tma3(&maps.m[RES::W2], full, dst, 0, 0, p0 + 2 * s, EVICT_FIRST);
        tma3(&maps.m[RES::W2SF], full, sf_dst, 0, 0, p0 + 2 * s, EVICT_FIRST);
      }
      if (slot != chk) {
        cnt_wait(g.cnt + RES::CHQ + slot, RES::MT13);
        asm volatile("fence.proxy.async.global;" ::: "memory");
        chk = slot;
      }
      if (na == 2) {
        tma3(&maps.m[RES::HQ_KT],
             full,
             dst + W_STAGE,
             0,
             slot * T,
             2 * s,
             EVICT_LAST);
        tma3(&maps.m[RES::HQ_SF2],
             full,
             hsf_dst,
             0,
             0,
             slot * KT2 + 2 * s,
             EVICT_LAST);
      } else {
        tma2(&maps.m[RES::HQ_A8],
             full,
             dst + W_STAGE,
             2 * s * 128,
             slot * T,
             EVICT_LAST);
        tma3(&maps.m[RES::HQ_SF],
             full,
             hsf_dst,
             0,
             0,
             slot * KT2 + 2 * s,
             EVICT_LAST);
      }
    }
    if (++st == SMAX) {
      st = 0;
      ph ^= 1u;
    }
  }
}

// issuer iss of a W2 tile (issue_job_warp's protocol: the stages with s % 2 ==
// iss into accumulator half iss of stage a; a whole warp, one elected lane
// issues), with the h_q scales of each (stage, piece) copied to TMEM too
template <class RES>
__device__ __forceinline__ void
    issue_w2_job(uint32_t base, uint32_t tb, int iss, int &g, int a) {
  constexpr int KT2 = RES::KT2, NST = RES::NST2;
  uint32_t const acc = tb + T * a + ACC_STAGES * T * iss; // (T = 8: 8 a + 32 iss)
  bool first = true;
  for (int s = 0; s < NST; s++, g++) {
    bool const mine = (s & 1) == iss;
    int const st = g % SMAX;
    mbar_wait(rt.full0 + 8 * st, (g / SMAX) & 1);
    fence_after();
    bool const one = issuer_lane<true>();
    if (!mine) {
      if (one) {
        mbar_arrive(rt.empty0 + 8 * st);
      }
      continue;
    }
    int const na = (KT2 - 2 * s < 2) ? KT2 - 2 * s : 2;
    for (int at = 0; at < na; at++) {
      uint32_t const sfa = tb + TM_W13_SF + 4 * (st * 2 + at),
                     sfb = tb + TM_HQ + 4 * (st * 2 + at);
      if (one) {
        cp_sf(sfa, base + OFF_WSF + (st * 2 + at) * SF_CHUNK);
        cp_sf(sfb, base + Smem<RES>::OFF_HSF + (st * 2 + at) * SF_CHUNK);
      }
      uint64_t const dw = mkdesc(base + st * FSTAGE + at * 16384);
      uint64_t const db = mkdesc(base + st * FSTAGE + W_STAGE + at * (T * 128));
      if (one) {
        mma_mx(acc, dw, db, idesc_mx(0), first ? 0u : 1u, sfa, sfb);
        mma_mx(acc, dw + 2, db + 2, idesc_mx(1), 1u, sfa, sfb);
        mma_mx(acc, dw + 4, db + 4, idesc_mx(2), 1u, sfa, sfb);
        mma_mx(acc, dw + 6, db + 6, idesc_mx(3), 1u, sfa, sfb);
      }
      first = false;
    }
    if (one) {
      tc_commit(rt.empty0 + 8 * st);
    }
  }
  if (issuer_lane<true>()) {
    tc_commit(rt.acc_full0 + 8 * a); // `acc_full` count 2: both issuers
  }
}

// all 8 warps of the SM, until the queue is empty (moe_experts.cuh
// run_expert_dynamic's roles: warp 0 lane 0 takes the entries and arms the
// even stages, warp 7 the odd ones, warps 1 and 6 issue, warps 2..5 drain).
// gl / gi0 / gi1: ring stage counters of the loaders / issuers; wt:
// accumulator tiles so far
template <class RES>
__device__ __forceinline__ void run_dynamic(G const &g,
                                            Maps const &maps,
                                            float *racc,
                                            uint32_t base,
                                            uint32_t tb,
                                            int &gl,
                                            int &gi0,
                                            int &gi1,
                                            int &wt STAGE_STAMP_PARAM) {
  constexpr int KT_LAT = RES::KT_LAT, MT13 = RES::MT13, OT2 = RES::OT2,
                K = RES::K, LAT = RES::LAT, IRP = RES::IRP, KT2 = RES::KT2;
  int *const eid_s = RES::eid(), *const ntok_s = RES::ntok(),
             *const tok_s = RES::tok(), *const start_s = RES::start();
  float *const w_s = RES::wt();
  unsigned char *const kk_s = RES::kk();
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  // the queue (moe_experts.cuh): W13 items, then W2 tiles n2 = slots x OT2 in
  // 2-tile entries, the last t1 = min(the tiles after the first ~4/5 of the
  // slots, NSM) in 1-tile entries
  int const n_live = s_ints[1], n13 = n_live * MT13, n2 = n_live * OT2;
  int const n4 = ((n2 * 4 / 5) / OT2) * OT2,
            t1 = (n2 - n4 < NSM) ? n2 - n4 : NSM, off1 = n2 - t1,
            n2e = (off1 + 1) / 2;
  int const nentries = n13 + n2e + t1;
  float *acc_s = reinterpret_cast<float *>(rt_sm() + Smem<RES>::OFF_ACC);
  auto entry_of = [&](int idx, QueueEntry &e) {
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
  auto w13_job = [&](QueueEntry const &e, TileJob &j) {
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
    j.amap = &maps.m[RES::ZQ];
    j.amap2 = &maps.m[RES::ZQ_KT];
    j.tile0 = j.chunk0 = ex * (MT13 * KT_LAT) + e.b * KT_LAT;
  };
  if (warp == 0) {
    if (lane == 0) {
      int qn = 0, chk = -1; // chk: the slot checked last (load_w2_job)
      bool maps_pf = false;
      auto publish = [&](QueueEntry const &e, int qn_) {
        int const qpos = qn_ % W2QD;
        if (qn_ >= W2QD) {
          mbar_wait(rt.qempty0 + 8 * qpos, ((qn_ / W2QD) - 1) & 1);
        }
        if (e.kind == QE_W2 && !maps_pf) { // the W2 maps -> descriptor cache
          prefetch_tmap(&maps.m[RES::W2X2]);
          prefetch_tmap(&maps.m[RES::W2]);
          prefetch_tmap(&maps.m[RES::W2SFX2]);
          prefetch_tmap(&maps.m[RES::W2SF]);
          prefetch_tmap(&maps.m[RES::HQ_KT]);
          prefetch_tmap(&maps.m[RES::HQ_SF2]);
          maps_pf = true;
        }
        w2q[qpos][0] = e.kind;
        w2q[qpos][1] = e.a;
        w2q[qpos][2] = e.b;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbar_arrive(rt.qfull0 + 8 * qpos);
        if (qn_ >= W2QD) {
          mbar_wait(rt.qempty7_0 + 8 * qpos, ((qn_ / W2QD) - 1) & 1);
        }
        w2q7[qpos][0] = e.kind;
        w2q7[qpos][1] = e.a;
        w2q7[qpos][2] = e.b;
        mbar_arrive(rt.qfull7_0 + 8 * qpos);
      };
      QueueEntry e;
      entry_of(s_ints[2], e);
      publish(e, qn);
      while (e.kind != QE_END) {
        QueueEntry en;
        if (e.kind == QE_W2) {
          int const idx1 = (int)atom_inc_lane0(g.cnt + RES::W2NEXT);
          int const slot = e.a / OT2, ex = eid_s[slot], ot0 = e.a - slot * OT2;
          for (int i = 0; i < e.b; i++) {
            load_w2_job<RES>(maps, g, base, gl, 0, slot, ex, ot0 + i, chk);
          }
          entry_of(idx1, en);
          publish(en, qn + 1);
        } else {
          TileJob j;
          w13_job(e, j);
          load_job(j, base, gl, 0, 0, W13_TAKE_AT);
          int const idx1 = (int)atom_inc_lane0(g.cnt + RES::W2NEXT);
          load_job(j, base, gl, 0, W13_TAKE_AT);
          entry_of(idx1, en);
          publish(en, qn + 1);
        }
        e = en;
        qn++;
      }
    }
  } else if (warp == 1 || warp == 6) { // the issuers
    int const iss = warp_same<true>((warp == 1) ? 0 : 1);
    int &gi = (warp == 1) ? gi0 : gi1;
    gi = warp_same<true>(gi);
    uint32_t const tbu = (uint32_t)warp_same<true>((int)tb),
                   baseu = (uint32_t)warp_same<true>((int)base);
    int wtt0 = warp_same<true>(wt);
    for (int qn = 0;; qn++) {
      int const qpos = qn % W2QD;
      mbar_wait(rt.qfull0 + 8 * qpos, (qn / W2QD) & 1);
      QueueEntry const e{warp_same<true>(w2q[qpos][0]),
                         warp_same<true>(w2q[qpos][1]),
                         warp_same<true>(w2q[qpos][2])};
      if (e.kind == QE_END) {
        break;
      }
      int const ntile = (e.kind == QE_W2) ? e.b : 1;
      for (int i = 0; i < ntile; i++) {
        int const wtt = wtt0 + i,
                  a = ACC_STAGES == 4 ? (wtt & 3) : acc_slot(wtt);
        if (wtt >= ACC_STAGES) {
          mbar_wait(rt.acc_empty0 + 8 * a,
                    ((ACC_STAGES == 4 ? (wtt >> 2) : acc_round(wtt)) - 1) & 1);
        }
        if (e.kind == QE_W2) {
          issue_w2_job<RES>(baseu, tbu, iss, gi, a);
        } else {
          TileJob j;
          w13_job(e, j);
          issue_job_warp(j,
                         baseu,
                         tbu,
                         iss,
                         gi,
                         a,
                         tbu + TM_ZQ STAGE_STAMP_ARG((iss == 0 && qn == 0) ? stamp
                                                                        : nullptr));
        }
      }
      wtt0 += ntile;
    }
  } else if (warp == 7) { // second loader: the odd stages
    int chk7 = -1;
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
        int const slot = e.a / OT2, ex = eid_s[slot], ot0 = e.a - slot * OT2;
        for (int i = 0; i < e.b; i++) {
          if (lane == 0) {
            load_w2_job<RES>(maps, g, base, gl, 1, slot, ex, ot0 + i, chk7);
          } else {
            gl += RES::NST2;
          }
        }
        __syncwarp();
      } else {
        TileJob j;
        w13_job(e, j);
        if (lane == 0) {
          load_job(j, base, gl, 1);
        } else {
          gl += j.nst;
        }
      }
    }
  } else if (warp >= 2 && warp <= 5) { // the epilogue, 128 threads
    float const lim = __int_as_float(RES::LIMIT_BITS);
    int wtt0 = wt;
    for (int qn = 0;; qn++) {
      int const qpos = qn % W2QD;
      mbar_wait(rt.qfull0 + 8 * qpos, (qn / W2QD) & 1);
      QueueEntry const e{w2q[qpos][0], w2q[qpos][1], w2q[qpos][2]};
      if (e.kind == QE_END) {
        break;
      }
      if (e.kind == QE_W2) { // R[(t, k)][128 ot + row] = weight x result
        for (int item = 0; item < e.b; item++) {
          int const wtt = wtt0 + item,
                    a = ACC_STAGES == 4 ? (wtt & 3) : acc_slot(wtt),
                    par = (ACC_STAGES == 4 ? (wtt >> 2) : acc_round(wtt)) & 1,
                    w = e.a + item, slot = w / OT2, ot = w - slot * OT2,
                    nt = ntok_s[slot];
          float v[T];
          int const row = drain_acc(tb, a, par, v); // both issuers' halves
          for (int i = 0; i < nt; i++) {
            int const e_ = start_s[slot] + i; // the slot's i-th routing entry
            int const t = tok_s[e_];
            float vt = v[0]; // v[t] by an unrolled select (no local memory)
#pragma unroll
            for (int k = 1; k < T; k++) {
              vt = (t == k) ? v[k] : vt;
            }
            racc[((size_t)t * K + kk_s[e_]) * LAT + ot * 128 + row] =
                w_s[e_] * vt;
          }
        }
        wtt0 += e.b;
      } else { // W13: rows 0..63 gate, 64..127 up of features [64 m, 64 m + 64)
        int const slot = e.a, mt = e.b,
                  a = ACC_STAGES == 4 ? (wtt0 & 3) : acc_slot(wtt0),
                  par = (ACC_STAGES == 4 ? (wtt0 >> 2) : acc_round(wtt0)) & 1;
        float v[T];
        int const row = drain_acc(tb, a, par, v);
        for (int t = 0; t < T; t++) {
          acc_s[row * T + t] = v[t];
        }
        asm volatile("bar.sync 1, 128;" ::: "memory");
        if (row < 64) { // a warp's 32 lanes: one 32-feature group of h
          int const f = mt * 64 + row, kt2 = f >> 7, kb = (f >> 5) & 3;
          // 8 tokens at a time (the registers of 8; T = 8: once, t0 = 0)
          auto quant8 = [&](int t0) {
            float h[8], am[8];
#pragma unroll
            for (int t = 0; t < 8; t++) {
              float const gt = fminf(bf16r(acc_s[row * T + t0 + t]), lim);
              float const up = fminf(
                  fmaxf(bf16r(acc_s[(64 + row) * T + t0 + t]), -lim), lim);
              h[t] = bf16r(gt / (1.0f + expf(-gt)) * up);
              am[t] = fabsf(h[t]);
            }
#pragma unroll
            for (int d = 16; d > 0; d >>= 1)
#pragma unroll
              for (int t = 0; t < 8; t++) {
                am[t] = fmaxf(am[t], __shfl_xor_sync(0xffffffffu, am[t], d));
              }
#pragma unroll
            for (int t = 0; t < 8; t++) {
              int const ex = xq_exponent(am[t]), tt = t0 + t;
              buf_at<uint8_t>(g, RES::HQ)[((size_t)slot * T + tt) * IRP + f] =
                  cvt_e4m3_nv(h[t] *
                              __uint_as_float((uint32_t)(127 - ex) << 23));
              if (lane == 0) { // row tt of the chunk: byte 16 (tt % 32) + 4 (tt
                               // / 32) + kb
                buf_at<uint8_t>(
                    g, RES::HSF)[((size_t)slot * KT2 + kt2) * SF_CHUNK +
                                 (tt % 32) * 16 + 4 * (tt / 32) + kb] =
                    (uint8_t)(ex + 127);
              }
            }
          };
          if constexpr (T == 8) {
            quant8(0);
          } else {
#pragma unroll 1
            for (int t0 = 0; t0 < T; t0 += 8) {
              quant8(t0);
            }
          }
        }
        // the bar.sync orders every thread's h_q / scale stores before thread
        // 64's red.release.gpu (as moe_experts.cuh)
        asm volatile("bar.sync 1, 128;" ::: "memory");
        if (threadIdx.x == 64) {
          cnt_add(g.cnt + RES::CHQ + slot, 1); // the W2 loaders acquire MT13
        }
        wtt0++;
      }
      mbar_arrive(rt.qempty0 + 8 * qpos); // 128 arrivals: entry retired
    }
  }
}

// the switch into the expert phase: fresh ring barriers and stage counters;
// x_q's KT_LAT scale chunks -> TMEM columns TM_ZQ.. (warp 1); the NPAIR routing
// pairs -> shared memory (warps 2..5, table_poll)
template <class RES>
__device__ __forceinline__ void phase_switch(unsigned long long const *pairs,
                                             Maps const &maps,
                                             uint32_t base,
                                             uint32_t tb,
                                             int &gl,
                                             int &gi0,
                                             int &gi1) {
  constexpr int KT_LAT = RES::KT_LAT, OFF_ZQ_SF = Smem<RES>::OFF_ZQ_SF;
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    ring_reinit(false);
  }
  gl = gi0 = gi1 = 0;
  __syncthreads();
  if (threadIdx.x == 0) {
    mbar_expect(rt.misc0, KT_LAT * SF_CHUNK);
    tma3(&maps.m[RES::ZQ_SF], rt.misc0, base + OFF_ZQ_SF, 0, 0, 0, EVICT_LAST);
  }
  mbar_wait(rt.misc0, 0);
  fence_after();
  if (warp == 1 && lane == 0) {
    for (int kt = 0; kt < KT_LAT; kt++) {
      cp_sf(tb + TM_ZQ + 4 * kt, base + OFF_ZQ_SF + kt * SF_CHUNK);
    }
    tc_commit(rt.misc0 + 16);
  }
  if (warp >= 2 && warp <= 5) { // one pair per epilogue thread (more: in turn)
    for (int i = threadIdx.x - 64; i < RES::NPAIR; i += 128) {
      table_poll<RES>(pairs, i);
    }
  }
  mbar_wait(rt.misc0 + 16, 0);
  fence_after();
  __syncthreads();
}

#ifdef MOE_CMP
// ---- (11-23, default off) -DMOE_CMP=16 at T >= 32: an expert slot's own
// tokens only. build_table's slots are cut into virtual slots of at most NC
// tokens (split_slots: vslot v = an expert and its tokens tok[start[v] ..
// + ntok[v])). A W13 item (v, m) and a W2 tile (v, ot) run MMA N = 8 ceil(n /
// 8) (n = ntok[v]) instead of N = T: the W13 stage holds the vslot's x_q rows
// (tile::gather4 of 4 rows, rows past n repeat the first token), the W2 stage
// the vslot's NC h_q rows (h_q row v NC + c: the vslot's c-th token). Its own
// ring (CSMAX stages of CFST bytes: 2 weight pieces + 2 K tiles x NC rows, in
// the layer ring's area) and barriers (cbars). B scales: W13 per stage from
// x_q's KT_LAT scale chunks kept in shared memory (OFF_ZSF; the owning issuer
// writes line c word 0 = token c's word into the stage's chunk, OFF_CBSF), W2
// by TMA (h_q's chunks, line c = place c). TMEM: accumulators T-wise as
// ACC_STAGES ticks of NC columns, the weight scales of (st, at) at TM_CW +
// 4 (2 st + at), the B scales at TM_CB + 4 (2 st + at). A token's results
// are those of the N = T build (its column's math does not depend on the
// others; the h_q / routed rows are the same bytes).
namespace cmp {
#ifndef MOE_CMP_MIN_T
#define MOE_CMP_MIN_T 32
#endif
constexpr int NC = MOE_CMP;
static_assert(NC == 16, "MOE_CMP: 16-token virtual slots");
constexpr bool ON = T >= MOE_CMP_MIN_T && T >= 16; // (-DMOE_CMP_MIN_T: 16 or 32)
#ifndef MOE_CMP_ST
#define MOE_CMP_ST 5
#endif
constexpr int CFST = W_STAGE + 2 * NC * 128, CSMAX = MOE_CMP_ST;
static_assert(CFST % 1024 == 0 && CSMAX >= 2, "MOE_CMP: 1024-B stages");
constexpr int TM_CW = 128, TM_CB = TM_CW + 8 * CSMAX;
static_assert(2 * ACC_STAGES * NC <= TM_CW && TM_CB + 8 * CSMAX <= 512,
              "MOE_CMP: TMEM columns");
__device__ __forceinline__ uint64_t *cbars() {
  __shared__ __align__(8) uint64_t b[2 * CSMAX];
  return b;
}
__device__ __forceinline__ uint32_t cfull(int st) {
  return su32(cbars() + st);
}
__device__ __forceinline__ uint32_t cempty(int st) {
  return su32(cbars() + CSMAX + st);
}
// kind::mxf8f6f4 e2m1 x e4m3, N = n (a multiple of 8), scale id sf
__device__ __forceinline__ uint32_t idesc_n(int n, uint32_t sf) {
  return idesc_mx((IDESC_MX & ~(0x3Fu << 17)) | ((uint32_t)(n / 8) << 17), sf);
}
// tile::gather4: rows r0..r3 (8 bits each in rows4) x the 128-B K tile at byte
// column c0 -> 4 x 128 B at sm (SW128 by address, as one box of rows would)
__device__ __forceinline__ void tma_gather4(void const *d,
                                            uint32_t mb,
                                            uint32_t sm,
                                            int c0,
                                            uint32_t rows4,
                                            uint64_t hint) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4."
               "mbarrier::complete_tx::bytes.L2::cache_hint"
               " [%0], [%1, {%3, %4, %5, %6, %7}], [%2], %8;" ::"r"(sm),
               "l"((uint64_t)d),
               "r"(mb),
               "r"(c0),
               "r"(rows4 & 0xffu),
               "r"((rows4 >> 8) & 0xffu),
               "r"((rows4 >> 16) & 0xffu),
               "r"(rows4 >> 24),
               "l"(hint)
               : "memory");
}
// shared memory (from the dynamic base): the stages (past the layer ring's
// area when CSMAX x CFST > OFF_WSF: only this task uses it then), 2 weight
// scale chunks a stage, 2 B scale chunks a stage, x_q's scales token-major
// (word t KT_LAT + kt: token t's 4 bytes of K tile kt), the W13 epilogue's
// 128 x NC fp32
template <class RES>
struct CSmem {
  static constexpr int OFF_CWSF = CSMAX * CFST,
                       OFF_CBSF = OFF_CWSF + CSMAX * 2 * SF_CHUNK,
                       OFF_ZSF = OFF_CBSF + CSMAX * 2 * SF_CHUNK,
                       OFF_CACC = OFF_ZSF + T * RES::KT_LAT * 4,
                       BYTES = OFF_CACC + 128 * NC * 4;
  static_assert(RES::KT_LAT * SF_CHUNK <= OFF_CWSF,
                "MOE_CMP: x_q's scale chunks staged in the stages' area");
};
// the maps the compact path adds (the node's map slots 15..17): x_q rows for
// tile::gather4 (box 128 B x 1 row), h_q 2 K tiles x NC rows, h_q 1 K tile x NC
// rows
template <class MAP_SLOTS>
struct CMaps {
  static constexpr int ZQG = slot_at<MAP_SLOTS, 15>(),
                       HQC_KT = slot_at<MAP_SLOTS, 16>(),
                       HQC_A8 = slot_at<MAP_SLOTS, 17>();
};

// after build_table (256 threads): slot s of n tokens -> ceil(n / NC) vslots
// in place (the same expert, NC consecutive entries each; the last the rest),
// s_ints[1] = the vslot count
template <class RES>
__device__ __forceinline__ void split_slots() {
  constexpr int PER = (RES::NSLOT + 255) / 256, K = RES::K;
  int *const eid_s = RES::eid(), *const ntok_s = RES::ntok(),
             *const start_s = RES::start();
  int const n_live = s_ints[1], warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  // 0. a slot's entries in token order, the same on every SM (build_table
  // places them by a shared atomic: an order that differs from SM to SM, while
  // a vslot's tokens and its h_q rows must agree across the SMs that run its
  // W13 items and W2 tiles): per slot a T-bit mask of its tokens (fill / cnt_e
  // as scratch: build_table is done with them), a pair's place = the bits
  // below its token
  {
    static_assert(T <= 64 && RES::NE >= RES::NSLOT, "MOE_CMP: 64-bit masks");
    unsigned *const mlo = reinterpret_cast<unsigned *>(RES::fill()),
                    *const mhi = reinterpret_cast<unsigned *>(RES::cnt_e());
    int *const sel_ss = RES::sel(), *const tok_s = RES::tok();
    float *const wsel_ss = RES::wsel(), *const w_s = RES::wt();
    unsigned char *const kk_s = RES::kk();
    for (int i = threadIdx.x; i < n_live; i += 256) {
      mlo[i] = 0u;
      mhi[i] = 0u;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < RES::NPAIR; i += 256) {
      int const t = i / K;
      atomicOr(t < 32 ? &mlo[sel_ss[i]] : &mhi[sel_ss[i]], 1u << (t & 31));
    }
    __syncthreads();
    for (int i = threadIdx.x; i < RES::NPAIR; i += 256) {
      int const s = sel_ss[i], t = i / K;
      unsigned const lo = mlo[s], hi = mhi[s];
      int const pos = t < 32 ? __popc(lo & ((1u << t) - 1u))
                             : __popc(lo) + __popc(hi & ((1u << (t - 32)) - 1u));
      int const e_ = start_s[s] + pos;
      tok_s[e_] = t;
      w_s[e_] = wsel_ss[i];
      kk_s[e_] = (unsigned char)(i % K);
    }
    __syncthreads();
  }
  int ex[PER], nt[PER], s0[PER], nv[PER], tot = 0;
#pragma unroll
  for (int q = 0; q < PER; q++) {
    int const i = threadIdx.x * PER + q;
    bool const live = i < n_live;
    ex[q] = live ? eid_s[i] : 0;
    nt[q] = live ? ntok_s[i] : 0;
    s0[q] = live ? start_s[i] : 0;
    nv[q] = (nt[q] + NC - 1) / NC;
    tot += nv[q];
  }
  int inc = tot;
#pragma unroll
  for (int o = 1; o < 32; o <<= 1) {
    int const u = __shfl_up_sync(0xffffffffu, inc, o);
    if (lane >= o) {
      inc += u;
    }
  }
  __syncthreads(); // every thread has read its slots (before any write)
  if (lane == 31) {
    s_ints[24 + warp] = inc;
  }
  __syncthreads();
  int base = inc - tot, all = 0;
  for (int w_ = 0; w_ < 8; w_++) {
    int const sw = s_ints[24 + w_];
    base += (w_ < warp) ? sw : 0;
    all += sw;
  }
#pragma unroll
  for (int q = 0; q < PER; q++) {
    for (int j = 0; j < nv[q]; j++, base++) {
      eid_s[base] = ex[q];
      start_s[base] = s0[q] + j * NC;
      ntok_s[base] = (nt[q] - j * NC < NC) ? nt[q] - j * NC : NC;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    s_ints[1] = all;
  }
  __syncthreads();
}

// W13 loader par (0: even stages, warp 0; 1: odd, warp 7) of item (v, mt),
// stages [s0, s1): the 2 weight pieces + their scale chunks, per K tile N / 4
// tile::gather4 of the vslot's x_q rows. g: this loader's stage counter
template <class RES, class CM>
__device__ __forceinline__ void load_w13c(Maps const &maps,
                                          uint32_t base,
                                          int &g,
                                          int par,
                                          int v,
                                          int mt,
                                          int s0,
                                          int s1) {
  constexpr int KT_LAT = RES::KT_LAT, NST = KT_LAT / 2;
  static_assert(KT_LAT % 2 == 0, "MOE_CMP: whole 2-piece W13 stages");
  int const n = RES::ntok()[v], N = (n + 7) & ~7, nq = N >> 2;
  int const *const tk = RES::tok() + RES::start()[v];
  uint32_t rows[NC / 4];
#pragma unroll
  for (int q = 0; q < NC / 4; q++) {
    uint32_t r = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int const c = 4 * q + i;
      r |= (uint32_t)tk[c < n ? c : 0] << (8 * i);
    }
    rows[q] = r;
  }
  int const tile0 = RES::eid()[v] * (RES::MT13 * KT_LAT) + mt * KT_LAT;
  int const send = NST < s1 ? NST : s1;
  int st = g % CSMAX;
  uint32_t ph = ((g / CSMAX) - 1) & 1;
  for (int s = s0; s < send; s++, g++) {
    if ((s & 1) == par) {
      if (g >= CSMAX) {
        mbar_wait(cempty(st), ph);
      }
      uint32_t const full = cfull(st), dst = base + st * CFST;
      mbar_expect(full, 2 * (8192 + SF_CHUNK) + 2 * N * 128);
      tma3(&maps.m[RES::W13X2], full, dst, 0, 0, tile0 + 2 * s, EVICT_FIRST);
      tma3(&maps.m[RES::W13SFX2],
           full,
           base + CSmem<RES>::OFF_CWSF + st * 2 * SF_CHUNK,
           0,
           0,
           tile0 + 2 * s,
           EVICT_FIRST);
#pragma unroll
      for (int at = 0; at < 2; at++) {
#pragma unroll
        for (int q = 0; q < NC / 4; q++) {
          if (q < nq) {
            tma_gather4(&maps.m[CM::ZQG],
                        full,
                        dst + W_STAGE + at * (NC * 128) + 512 * q,
                        (2 * s + at) * 128,
                        rows[q],
                        EVICT_LAST);
          }
        }
      }
    }
    if (++st == CSMAX) {
      st = 0;
      ph ^= 1u;
    }
  }
}

// W2 loader par of tile (v, ot): load_w2_job's stages on the compact ring, the
// vslot's NC h_q rows
template <class RES, class CM>
__device__ __forceinline__ void load_w2c(Maps const &maps,
                                         G const &g,
                                         uint32_t base,
                                         int &gq,
                                         int par,
                                         int v,
                                         int ot,
                                         int &chk) {
  constexpr int KT2 = RES::KT2, NST = RES::NST2;
  int st = gq % CSMAX;
  uint32_t ph = ((gq / CSMAX) - 1) & 1;
  int const p0 = RES::eid()[v] * (RES::OT2 * KT2) + ot * KT2;
  for (int s = 0; s < NST; s++, gq++) {
    if ((s & 1) == par) {
      if (gq >= CSMAX) {
        mbar_wait(cempty(st), ph);
      }
      int const na = (KT2 - 2 * s < 2) ? KT2 - 2 * s : 2;
      uint32_t const full = cfull(st), dst = base + st * CFST,
                     sf_dst = base + CSmem<RES>::OFF_CWSF + st * 2 * SF_CHUNK,
                     hsf_dst = base + CSmem<RES>::OFF_CBSF + st * 2 * SF_CHUNK;
      mbar_expect(full, na * (8192 + SF_CHUNK + NC * 128 + SF_CHUNK));
      if (na == 2) {
        tma3(&maps.m[RES::W2X2], full, dst, 0, 0, p0 + 2 * s, EVICT_FIRST);
        tma3(&maps.m[RES::W2SFX2], full, sf_dst, 0, 0, p0 + 2 * s, EVICT_FIRST);
      } else {
        tma3(&maps.m[RES::W2], full, dst, 0, 0, p0 + 2 * s, EVICT_FIRST);
        tma3(&maps.m[RES::W2SF], full, sf_dst, 0, 0, p0 + 2 * s, EVICT_FIRST);
      }
      if (v != chk) {
        cnt_wait(g.cnt + RES::CHQ + v, RES::MT13);
        asm volatile("fence.proxy.async.global;" ::: "memory");
        chk = v;
      }
      if (na == 2) {
        tma3(&maps.m[CM::HQC_KT],
             full,
             dst + W_STAGE,
             0,
             v * NC,
             2 * s,
             EVICT_LAST);
        tma3(&maps.m[RES::HQ_SF2],
             full,
             hsf_dst,
             0,
             0,
             v * KT2 + 2 * s,
             EVICT_LAST);
      } else {
        tma2(&maps.m[CM::HQC_A8],
             full,
             dst + W_STAGE,
             2 * s * 128,
             v * NC,
             EVICT_LAST);
        tma3(&maps.m[RES::HQ_SF],
             full,
             hsf_dst,
             0,
             0,
             v * KT2 + 2 * s,
             EVICT_LAST);
      }
    }
    if (++st == CSMAX) {
      st = 0;
      ph ^= 1u;
    }
  }
}

// issuer iss (a whole warp, one elected lane issues) of a W13 item (v, mt) or a
// W2 tile (v, ot): its stages with s % 2 == iss into accumulator half iss of
// tick a, N = 8 ceil(ntok[v] / 8). W13: before a stage's MMAs, lanes 0..15
// (at 0) and 16..31 (at 1) write line c = lane % 16 word 0 of the stage's B
// scale chunk: token c's x_q scale word of K tile 2 s + at (rows past n: the
// first token's)
template <class RES, bool W13>
__device__ __forceinline__ void
    issue_c(uint32_t base, uint32_t tb, int iss, int &g, int a, int v) {
  constexpr int NST = W13 ? RES::KT_LAT / 2 : RES::NST2, KT2 = RES::KT2;
  int const n = RES::ntok()[v], N = (n + 7) & ~7, lane = threadIdx.x & 31,
            c = lane & 15, atl = lane >> 4;
  int const tc = RES::tok()[RES::start()[v] + (c < n ? c : 0)];
  uint32_t const zsrc = base + CSmem<RES>::OFF_ZSF + 4 * (tc * RES::KT_LAT);
  uint32_t const acc = tb + NC * a + ACC_STAGES * NC * iss;
  bool first = true;
  for (int s = 0; s < NST; s++, g++) {
    bool const mine = (s & 1) == iss;
    int const st = g % CSMAX;
    mbar_wait(cfull(st), (g / CSMAX) & 1);
    fence_after();
    if (!mine) {
      if (issuer_lane<true>()) {
        mbar_arrive(cempty(st));
      }
      continue;
    }
    if constexpr (W13) {
      uint32_t w;
      asm volatile("ld.shared.u32 %0, [%1];"
                   : "=r"(w)
                   : "r"(zsrc + 4 * (2 * s + atl)));
      asm volatile("st.shared.u32 [%0], %1;" ::"r"(
                       base + CSmem<RES>::OFF_CBSF +
                       (st * 2 + atl) * SF_CHUNK + 16 * c),
                   "r"(w)
                   : "memory");
      asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    }
    bool const one = issuer_lane<true>();
    int const na = W13 ? 2 : ((KT2 - 2 * s < 2) ? KT2 - 2 * s : 2);
    for (int at = 0; at < na; at++) {
      uint32_t const sfa = tb + TM_CW + 4 * (st * 2 + at),
                     sfb = tb + TM_CB + 4 * (st * 2 + at);
      if (one) {
        cp_sf(sfa, base + CSmem<RES>::OFF_CWSF + (st * 2 + at) * SF_CHUNK);
        cp_sf(sfb, base + CSmem<RES>::OFF_CBSF + (st * 2 + at) * SF_CHUNK);
      }
      uint64_t const dw = mkdesc(base + st * CFST + at * 16384);
      uint64_t const db = mkdesc(base + st * CFST + W_STAGE + at * (NC * 128));
      if (one) {
        mma_mx(acc, dw, db, idesc_n(N, 0), first ? 0u : 1u, sfa, sfb);
        mma_mx(acc, dw + 2, db + 2, idesc_n(N, 1), 1u, sfa, sfb);
        mma_mx(acc, dw + 4, db + 4, idesc_n(N, 2), 1u, sfa, sfb);
        mma_mx(acc, dw + 6, db + 6, idesc_n(N, 3), 1u, sfa, sfb);
      }
      first = false;
    }
    if (one) {
      tc_commit(cempty(st));
    }
  }
  if (issuer_lane<true>()) {
    tc_commit(rt.acc_full0 + 8 * a); // `acc_full` count 2: both issuers
  }
}

// epilogue: wait for tick a, add the two issuers' halves -> v[NC] of this
// thread's row (TMEM lane)
__device__ __forceinline__ int drain_c(uint32_t tb, int a, int parity, float *v) {
  int const lgrp = (threadIdx.x >> 5) & 3, lane = threadIdx.x & 31;
  mbar_wait(rt.acc_full0 + 8 * a, parity);
  fence_after();
  float v2[NC];
  tmem_ld_cols_nw<NC>(tb + NC * a + ((uint32_t)(lgrp * 32) << 16), v);
  tmem_ld_cols_nw<NC>(tb + NC * a + ACC_STAGES * NC + ((uint32_t)(lgrp * 32) << 16),
                      v2);
  tmem_keep_cols<NC>(v);
  tmem_keep_cols<NC>(v2);
#pragma unroll
  for (int c = 0; c < NC; c++) {
    v[c] += v2[c];
  }
  fence_before();
  mbar_arrive(rt.acc_empty0 + 8 * a); // 128 arrivals
  return lgrp * 32 + lane;
}

// the phase switch of the compact path: fresh ring barriers (the layer's and
// cbars), x_q's KT_LAT scale chunks staged in the (idle) stages' area, then
// token-major -> OFF_ZSF (kept for the phase), the routing pairs
template <class RES>
__device__ __forceinline__ void phase_switch_c(unsigned long long const *pairs,
                                               Maps const &maps,
                                               uint32_t base,
                                               int &gl,
                                               int &gi0,
                                               int &gi1) {
  if (threadIdx.x == 0) {
    ring_reinit(false);
    for (int st = 0; st < CSMAX; st++) {
      mbar_init(cfull(st), 1);
      mbar_init(cempty(st), 2);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  gl = gi0 = gi1 = 0;
  __syncthreads();
  if (threadIdx.x == 0) {
    mbar_expect(rt.misc0, RES::KT_LAT * SF_CHUNK);
    tma3(&maps.m[RES::ZQ_SF], rt.misc0, base, 0, 0, 0, EVICT_LAST);
  }
  int const warp = threadIdx.x >> 5;
  if (warp >= 2 && warp <= 5) {
    for (int i = threadIdx.x - 64; i < RES::NPAIR; i += 128) {
      table_poll<RES>(pairs, i);
    }
  }
  mbar_wait(rt.misc0, 0);
  fence_after();
  { // chunk kt line t % 32 word t / 32 -> word t KT_LAT + kt
    char *const sm = rt_sm();
    for (int i = threadIdx.x; i < T * RES::KT_LAT; i += 256) {
      int const t = i / RES::KT_LAT, kt = i - t * RES::KT_LAT;
      *reinterpret_cast<uint32_t *>(sm + CSmem<RES>::OFF_ZSF + 4 * i) =
          *reinterpret_cast<uint32_t const *>(sm + kt * SF_CHUNK +
                                              16 * (t % 32) + 4 * (t / 32));
    }
  }
  // every thread's reads of the staged chunks before the ring's TMA writes
  // there (generic -> async proxy)
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
}

// run_dynamic's roles and queue on the compact ring (the vslots of
// split_slots): warp 0 lane 0 takes the entries and arms the even stages, warp
// 7 lane 0 the odd ones, warps 1 and 6 issue, warps 2..5 drain
template <class RES, class CM>
__device__ __forceinline__ void run_dynamic_c(G const &g,
                                              Maps const &maps,
                                              float *racc,
                                              uint32_t base,
                                              uint32_t tb,
                                              int &wt) {
  constexpr int MT13 = RES::MT13, OT2 = RES::OT2, K = RES::K, LAT = RES::LAT,
                IRP = RES::IRP, KT2 = RES::KT2;
  int *const ntok_s = RES::ntok(), *const tok_s = RES::tok(),
             *const start_s = RES::start();
  float *const w_s = RES::wt();
  unsigned char *const kk_s = RES::kk();
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int const n_live = s_ints[1], n13 = n_live * MT13, n2 = n_live * OT2;
  int const n4 = ((n2 * 4 / 5) / OT2) * OT2,
            t1 = (n2 - n4 < NSM) ? n2 - n4 : NSM, off1 = n2 - t1,
            n2e = (off1 + 1) / 2;
  int const nentries = n13 + n2e + t1;
  float *acc_s =
      reinterpret_cast<float *>(rt_sm() + CSmem<RES>::OFF_CACC);
  auto entry_of = [&](int idx, QueueEntry &e) {
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
  if (warp == 0) {
    if (lane == 0) {
      int qn = 0, chk = -1, gl = 0;
      bool maps_pf = false;
      auto publish = [&](QueueEntry const &e, int qn_) {
        int const qpos = qn_ % W2QD;
        if (qn_ >= W2QD) {
          mbar_wait(rt.qempty0 + 8 * qpos, ((qn_ / W2QD) - 1) & 1);
        }
        if (e.kind == QE_W2 && !maps_pf) {
          prefetch_tmap(&maps.m[RES::W2X2]);
          prefetch_tmap(&maps.m[RES::W2]);
          prefetch_tmap(&maps.m[RES::W2SFX2]);
          prefetch_tmap(&maps.m[RES::W2SF]);
          prefetch_tmap(&maps.m[CM::HQC_KT]);
          prefetch_tmap(&maps.m[CM::HQC_A8]);
          prefetch_tmap(&maps.m[RES::HQ_SF2]);
          prefetch_tmap(&maps.m[RES::HQ_SF]);
          maps_pf = true;
        }
        w2q[qpos][0] = e.kind;
        w2q[qpos][1] = e.a;
        w2q[qpos][2] = e.b;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbar_arrive(rt.qfull0 + 8 * qpos);
        if (qn_ >= W2QD) {
          mbar_wait(rt.qempty7_0 + 8 * qpos, ((qn_ / W2QD) - 1) & 1);
        }
        w2q7[qpos][0] = e.kind;
        w2q7[qpos][1] = e.a;
        w2q7[qpos][2] = e.b;
        mbar_arrive(rt.qfull7_0 + 8 * qpos);
      };
      prefetch_tmap(&maps.m[CM::ZQG]);
      QueueEntry e;
      entry_of(s_ints[2], e);
      publish(e, qn);
      while (e.kind != QE_END) {
        QueueEntry en;
        if (e.kind == QE_W2) {
          int const idx1 = (int)atom_inc_lane0(g.cnt + RES::W2NEXT);
          int const v = e.a / OT2, ot0 = e.a - v * OT2;
          for (int i = 0; i < e.b; i++) {
            load_w2c<RES, CM>(maps, g, base, gl, 0, v, ot0 + i, chk);
          }
          entry_of(idx1, en);
          publish(en, qn + 1);
        } else {
          load_w13c<RES, CM>(maps, base, gl, 0, e.a, e.b, 0, W13_TAKE_AT);
          int const idx1 = (int)atom_inc_lane0(g.cnt + RES::W2NEXT);
          load_w13c<RES, CM>(maps, base, gl, 0, e.a, e.b, W13_TAKE_AT, 1 << 20);
          entry_of(idx1, en);
          publish(en, qn + 1);
        }
        e = en;
        qn++;
      }
    }
  } else if (warp == 1 || warp == 6) { // the issuers
    int const iss = warp_same<true>((warp == 1) ? 0 : 1);
    int gi = 0;
    uint32_t const tbu = (uint32_t)warp_same<true>((int)tb),
                   baseu = (uint32_t)warp_same<true>((int)base);
    int wtt0 = warp_same<true>(wt);
    for (int qn = 0;; qn++) {
      int const qpos = qn % W2QD;
      mbar_wait(rt.qfull0 + 8 * qpos, (qn / W2QD) & 1);
      QueueEntry const e{warp_same<true>(w2q[qpos][0]),
                         warp_same<true>(w2q[qpos][1]),
                         warp_same<true>(w2q[qpos][2])};
      if (e.kind == QE_END) {
        break;
      }
      int const ntile = (e.kind == QE_W2) ? e.b : 1;
      for (int i = 0; i < ntile; i++) {
        int const wtt = wtt0 + i, a = acc_slot(wtt);
        if (wtt >= ACC_STAGES) {
          mbar_wait(rt.acc_empty0 + 8 * a, (acc_round(wtt) - 1) & 1);
        }
        if (e.kind == QE_W2) {
          issue_c<RES, false>(baseu, tbu, iss, gi, a, (e.a + i) / OT2);
        } else {
          issue_c<RES, true>(baseu, tbu, iss, gi, a, e.a);
        }
      }
      wtt0 += ntile;
    }
  } else if (warp == 7) { // second loader: the odd stages
    int chk7 = -1, gl = 0;
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
        int const v = e.a / OT2, ot0 = e.a - v * OT2;
        for (int i = 0; i < e.b; i++) {
          if (lane == 0) {
            load_w2c<RES, CM>(maps, g, base, gl, 1, v, ot0 + i, chk7);
          } else {
            gl += RES::NST2;
          }
        }
        __syncwarp();
      } else {
        if (lane == 0) {
          load_w13c<RES, CM>(maps, base, gl, 1, e.a, e.b, 0, 1 << 20);
        } else {
          gl += RES::KT_LAT / 2;
        }
      }
    }
  } else if (warp >= 2 && warp <= 5) { // the epilogue, 128 threads
    float const lim = __int_as_float(RES::LIMIT_BITS);
    int wtt0 = wt;
    for (int qn = 0;; qn++) {
      int const qpos = qn % W2QD;
      mbar_wait(rt.qfull0 + 8 * qpos, (qn / W2QD) & 1);
      QueueEntry const e{w2q[qpos][0], w2q[qpos][1], w2q[qpos][2]};
      if (e.kind == QE_END) {
        break;
      }
      if (e.kind == QE_W2) { // R[(t, k)][128 ot + row] = weight x result
        for (int item = 0; item < e.b; item++) {
          int const wtt = wtt0 + item, w = e.a + item, v = w / OT2,
                    ot = w - v * OT2, nt = ntok_s[v];
          float val[NC];
          int const row = drain_c(tb, acc_slot(wtt), acc_round(wtt) & 1, val);
          for (int i = 0; i < nt; i++) {
            int const e_ = start_s[v] + i; // the vslot's i-th entry
            float vi = val[0]; // val[i] by an unrolled select
#pragma unroll
            for (int k = 1; k < NC; k++) {
              vi = (i == k) ? val[k] : vi;
            }
            racc[((size_t)tok_s[e_] * K + kk_s[e_]) * LAT + ot * 128 + row] =
                w_s[e_] * vi;
          }
        }
        wtt0 += e.b;
      } else { // W13: rows 0..63 gate, 64..127 up of features [64 m, 64 m + 64)
        int const v = e.a, mt = e.b, N = (ntok_s[v] + 7) & ~7;
        float val[NC];
        int const row = drain_c(tb, acc_slot(wtt0), acc_round(wtt0) & 1, val);
#pragma unroll
        for (int c = 0; c < NC; c++) {
          acc_s[row * NC + c] = val[c];
        }
        asm volatile("bar.sync 1, 128;" ::: "memory");
        if (row < 64) { // a warp's 32 lanes: one 32-feature group of h
          int const f = mt * 64 + row, kt2 = f >> 7, kb = (f >> 5) & 3;
#pragma unroll 1
          for (int c0 = 0; c0 < N; c0 += 8) { // 8 places at a time
            float h[8], am[8];
#pragma unroll
            for (int t = 0; t < 8; t++) {
              float const gt = fminf(bf16r(acc_s[row * NC + c0 + t]), lim);
              float const up = fminf(
                  fmaxf(bf16r(acc_s[(64 + row) * NC + c0 + t]), -lim), lim);
              h[t] = bf16r(gt / (1.0f + expf(-gt)) * up);
              am[t] = fabsf(h[t]);
            }
#pragma unroll
            for (int d = 16; d > 0; d >>= 1)
#pragma unroll
              for (int t = 0; t < 8; t++) {
                am[t] = fmaxf(am[t], __shfl_xor_sync(0xffffffffu, am[t], d));
              }
#pragma unroll
            for (int t = 0; t < 8; t++) {
              int const ex = xq_exponent(am[t]), cc = c0 + t;
              buf_at<uint8_t>(g, RES::HQ)[((size_t)v * NC + cc) * IRP + f] =
                  cvt_e4m3_nv(h[t] *
                              __uint_as_float((uint32_t)(127 - ex) << 23));
              if (lane == 0) { // line cc of the chunk, word 0: byte 16 cc + kb
                buf_at<uint8_t>(
                    g, RES::HSF)[((size_t)v * KT2 + kt2) * SF_CHUNK + cc * 16 +
                                 kb] = (uint8_t)(ex + 127);
              }
            }
          }
        }
        // the bar.sync orders every thread's h_q / scale stores before thread
        // 64's red.release.gpu (as moe_experts.cuh)
        asm volatile("bar.sync 1, 128;" ::: "memory");
        if (threadIdx.x == 64) {
          cnt_add(g.cnt + RES::CHQ + v, 1); // the W2 loaders acquire MT13
        }
        wtt0++;
      }
      mbar_arrive(rt.qempty0 + 8 * qpos); // 128 arrivals: entry retired
    }
  }
}
} // namespace cmp
#endif // MOE_CMP

} // namespace swiglu_moe

// the node's Slots: its buffer slots (h_q, its scales), its map slots (w13,
// w13 x2, w13 scales, x2, w2, w2 x2, w2 scales, x2, x_q, x_q x2 K tiles, x_q's
// scale chunks, h_q x2 K tiles, h_q's scales, h_q one K tile, h_q's scales x2),
// its counter, the sizes: NE and K from topk_route's params (PAIRS), LAT from
// act_quant's (XQ), I and L its own params
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class XQ,
          class PAIRS>
using SwigluSlotsOf = swiglu_moe::Slots<slot_at<BUF_SLOTS, 0>(),
                                        slot_at<BUF_SLOTS, 1>(),
                                        SELF::counter,
                                        slot_at<MAP_SLOTS, 0>(),
                                        slot_at<MAP_SLOTS, 1>(),
                                        slot_at<MAP_SLOTS, 2>(),
                                        slot_at<MAP_SLOTS, 3>(),
                                        slot_at<MAP_SLOTS, 4>(),
                                        slot_at<MAP_SLOTS, 5>(),
                                        slot_at<MAP_SLOTS, 6>(),
                                        slot_at<MAP_SLOTS, 7>(),
                                        slot_at<MAP_SLOTS, 8>(),
                                        slot_at<MAP_SLOTS, 9>(),
                                        slot_at<MAP_SLOTS, 10>(),
                                        slot_at<MAP_SLOTS, 11>(),
                                        slot_at<MAP_SLOTS, 12>(),
                                        slot_at<MAP_SLOTS, 13>(),
                                        slot_at<MAP_SLOTS, 14>(),
                                        PAIRS::v[0],
                                        PAIRS::v[1],
                                        XQ::v[0],
                                        PARAMS::v[0],
                                        PARAMS::v[1]>;

// swiglu_moe_experts task (one per SM, task x on SM x): wait for x_q (its
// node's counter: all of act_quant's tasks), phase switch, routing table, then
// the queue -> the routed rows [T][K][LAT] in the node's output buffer; its
// counter counts the SMs done. PARAMS {I, L as fp32 bits}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class XQ,
          class PAIRS,
          class... REST>
__device__ __forceinline__ void run_swiglu_moe_experts(Maps const &maps,
                                                       G const &g,
                                                       KernelLocals &L,
                                                       StaticTask const &) {
  static_assert(SELF::buf >= 0 && SELF::counter >= 0 && PAIRS::buf >= 0 &&
                    param_or0<PAIRS, 6>() == 1,
                "swiglu_moe_experts: its output rows, its counter, the pairs "
                "of topk_route's DeepSeek-V4 gate (sorted by expert id)");
  static_assert(XQ::buf >= 0 && exchange_bytes[XQ::buf] == 0 &&
                    XQ::counter >= 0,
                "swiglu_moe_experts: x_q, act_quant's node buffer");
  using RES = SwigluSlotsOf<SELF, PARAMS, BUF_SLOTS, MAP_SLOTS, XQ, PAIRS>;
  int const sm_id = L.sm_id;
  __syncthreads(); // every warp is done with the placed tiles
  uint32_t pre_idx = 0;
  if (threadIdx.x == 0) {
    pre_idx = atom_inc_lane0(g.cnt + RES::W2NEXT);
  }
  for (int i = threadIdx.x; i < RES::NE; i += 256) {
    RES::cnt_e()[i] = 0;
  }
  if (threadIdx.x == 0) { // x_q: all act_quant tasks (acquire)
    cnt_wait(g.cnt + XQ::counter, XQ::x * XQ::y * XQ::z);
  }
  __syncthreads();
  asm volatile("fence.proxy.async.global;" ::: "memory"); // TMA reads x_q
  if (threadIdx.x == 0) {
    g.stamps[sm_id * NSTAMP + STAMP_TASK0] = gtime();
  }
#ifdef MOE_CMP
  constexpr bool CMP = swiglu_moe::cmp::ON;
#endif
  if (!L.ring_relaid) {
#ifdef MOE_CMP
    if constexpr (CMP) {
      swiglu_moe::cmp::phase_switch_c<RES>(
          buf_at<unsigned long long const>(g, PAIRS::buf),
          maps,
          L.base,
          L.gl,
          L.gi0,
          L.gi1);
    } else
#endif
    {
      swiglu_moe::phase_switch<RES>(
          buf_at<unsigned long long const>(g, PAIRS::buf),
          maps,
          L.base,
          L.tb,
          L.gl,
          L.gi0,
          L.gi1);
    }
    L.ring_relaid = true;
  }
#if defined(STATIC_TIMING_BUILD) && defined(MOE_FRONT_STAMPS)
  // (measurement only, 11-18) this task's 2 stage stamps: the phase switch done
  // (the routing pairs in shared memory) and the table built, instead of the
  // GEMM pipeline's first load issued / first stage landed
  if (threadIdx.x == 0 && L.stage_stamps) {
    L.stage_stamps[0] = gtime();
  }
#endif
  build_table<RES>();
#if defined(STATIC_TIMING_BUILD) && defined(MOE_FRONT_STAMPS)
  if (threadIdx.x == 0 && L.stage_stamps) {
    L.stage_stamps[1] = gtime();
  }
#define MOE_RD_STAMPS nullptr
#else
#define MOE_RD_STAMPS L.stage_stamps
#endif
#ifdef MOE_CMP
  if constexpr (CMP) {
    swiglu_moe::cmp::split_slots<RES>();
  }
#endif
  if (threadIdx.x == 0) {
    s_ints[2] = (int)pre_idx;
  }
  __syncthreads();
#ifdef MOE_CMP
  if constexpr (CMP) {
    swiglu_moe::cmp::run_dynamic_c<RES, swiglu_moe::cmp::CMaps<MAP_SLOTS>>(
        g, maps, buf_at<float>(g, SELF::buf), L.base, L.tb, L.wt);
  } else
#endif
  {
    swiglu_moe::run_dynamic<RES>(g,
                                 maps,
                                 buf_at<float>(g, SELF::buf),
                                 L.base,
                                 L.tb,
                                 L.gl,
                                 L.gi0,
                                 L.gi1,
                                 L.wt STAGE_STAMP_ARG(MOE_RD_STAMPS));
  }
#undef MOE_RD_STAMPS
  if (threadIdx.x == 0) {
    g.stamps[sm_id * NSTAMP + STAMP_TASK1] = gtime();
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(g.cnt + SELF::counter, 1); // this SM's routed rows are written
  }
}

// dynamic shared memory: the ring, then Smem's
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class XQ,
          class PAIRS,
          class... REST>
constexpr int smem_swiglu_moe_experts() {
  using RES = SwigluSlotsOf<SELF, PARAMS, BUF_SLOTS, MAP_SLOTS, XQ, PAIRS>;
#ifdef MOE_CMP
  if constexpr (swiglu_moe::cmp::ON) {
    return swiglu_moe::Smem<RES>::BYTES > swiglu_moe::cmp::CSmem<RES>::BYTES
               ? swiglu_moe::Smem<RES>::BYTES
               : swiglu_moe::cmp::CSmem<RES>::BYTES;
  }
#endif
  return swiglu_moe::Smem<RES>::BYTES;
}

} // namespace static_mk
