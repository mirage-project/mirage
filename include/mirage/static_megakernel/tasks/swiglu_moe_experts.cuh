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
// stage st, piece at, 272 + 4 (2 st + at) a W2 stage's h_q scales.
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
template <class RES>
struct Smem {
  static constexpr int OFF_ZQ_SF = OFF_W,          // x_q's scale chunks, staged
                                                    // once at the phase switch
      OFF_HSF = RING_BYTES,                        // SMAX x 2 h_q scale chunks
      OFF_ACC = OFF_HSF + SMAX * 2 * SF_CHUNK;     // a W13 item's accumulator
  static constexpr int BYTES = OFF_ACC + 128 * T * 4;
  static_assert(RES::KT_LAT * SF_CHUNK <= OFF_WSF,
                "the x_q scale chunks in the ring area");
  static_assert(72 + 4 * RES::KT_LAT <= 232 && 272 + 4 * 2 * SMAX <= 512,
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
  uint32_t const acc = tb + 8 * a + 32 * iss;
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
      uint32_t const sfa = tb + 232 + 4 * (st * 2 + at),
                     sfb = tb + 272 + 4 * (st * 2 + at);
      if (one) {
        cp_sf(sfa, base + OFF_WSF + (st * 2 + at) * SF_CHUNK);
        cp_sf(sfb, base + Smem<RES>::OFF_HSF + (st * 2 + at) * SF_CHUNK);
      }
      uint64_t const dw = mkdesc(base + st * FSTAGE + at * 16384);
      uint64_t const db = mkdesc(base + st * FSTAGE + W_STAGE + at * 1024);
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
        int const wtt = wtt0 + i, a = wtt & 3;
        if (wtt >= 4) {
          mbar_wait(rt.acc_empty0 + 8 * a, ((wtt >> 2) - 1) & 1);
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
                         tbu + 72 STAGE_STAMP_ARG((iss == 0 && qn == 0) ? stamp
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
          int const wtt = wtt0 + item, a = wtt & 3, par = (wtt >> 2) & 1,
                    w = e.a + item, slot = w / OT2, ot = w - slot * OT2,
                    nt = ntok_s[slot];
          float v[8];
          int const row = drain_acc(tb, a, par, v); // both issuers' halves
          for (int i = 0; i < nt; i++) {
            int const e_ = start_s[slot] + i; // the slot's i-th routing entry
            int const t = tok_s[e_];
            float vt = v[0]; // v[t] by an unrolled select (no local memory)
#pragma unroll
            for (int k = 1; k < 8; k++) {
              vt = (t == k) ? v[k] : vt;
            }
            racc[((size_t)t * K + kk_s[e_]) * LAT + ot * 128 + row] =
                w_s[e_] * vt;
          }
        }
        wtt0 += e.b;
      } else { // W13: rows 0..63 gate, 64..127 up of features [64 m, 64 m + 64)
        int const slot = e.a, mt = e.b, a = wtt0 & 3, par = (wtt0 >> 2) & 1;
        float v[8];
        int const row = drain_acc(tb, a, par, v);
        for (int t = 0; t < T; t++) {
          acc_s[row * 8 + t] = v[t];
        }
        asm volatile("bar.sync 1, 128;" ::: "memory");
        if (row < 64) { // a warp's 32 lanes: one 32-feature group of h
          int const f = mt * 64 + row, kt2 = f >> 7, kb = (f >> 5) & 3;
          float h[T], am[T];
#pragma unroll
          for (int t = 0; t < T; t++) {
            float const gt = fminf(bf16r(acc_s[row * 8 + t]), lim);
            float const up =
                fminf(fmaxf(bf16r(acc_s[(64 + row) * 8 + t]), -lim), lim);
            h[t] = bf16r(gt / (1.0f + expf(-gt)) * up);
            am[t] = fabsf(h[t]);
          }
#pragma unroll
          for (int d = 16; d > 0; d >>= 1)
#pragma unroll
            for (int t = 0; t < T; t++) {
              am[t] = fmaxf(am[t], __shfl_xor_sync(0xffffffffu, am[t], d));
            }
#pragma unroll
          for (int t = 0; t < T; t++) {
            int const ex = xq_exponent(am[t]);
            buf_at<uint8_t>(g, RES::HQ)[((size_t)slot * T + t) * IRP + f] =
                cvt_e4m3_nv(h[t] *
                            __uint_as_float((uint32_t)(127 - ex) << 23));
            if (lane == 0) {
              buf_at<uint8_t>(g,
                              RES::HSF)[((size_t)slot * KT2 + kt2) * SF_CHUNK +
                                        t * 16 + kb] = (uint8_t)(ex + 127);
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
// x_q's KT_LAT scale chunks -> TMEM columns 72.. (warp 1); the NPAIR routing
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
      cp_sf(tb + 72 + 4 * kt, base + OFF_ZQ_SF + kt * SF_CHUNK);
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
  if (!L.ring_relaid) {
    swiglu_moe::phase_switch<RES>(
        buf_at<unsigned long long const>(g, PAIRS::buf),
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
  swiglu_moe::run_dynamic<RES>(g,
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
  return swiglu_moe::Smem<
      SwigluSlotsOf<SELF, PARAMS, BUF_SLOTS, MAP_SLOTS, XQ, PAIRS>>::BYTES;
}

} // namespace static_mk
