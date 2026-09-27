// expert_queue.cuh -- the routed experts and the shared down GEMM, after routing. Every SM takes entries from one queue (the counter
// C_W2NEXT: an atomic add returns the next entry index), in this order:
//   1. W13 items: (expert slot, 64-feature tile m): gate and up rows of the expert x z_q (MXFP4 x e4m3, K = LAT), then
//      h_q = MXFP8(SiTU(gate, up)) for the tokens of that expert, then counter C_HQ[slot] += 1
//   2. W2 chunks: up to W2_CHUNK consecutive items (slot, 128-row output tile) of one expert: W2 x h_q (K = IR), then
//      R[t] += weight(t, expert) * result for the expert's tokens (red.add into Racc)
//   3. shared down entries: (128-row tile of S, K half): shared down weight x h_s, red.add into Sout
// Roles: warp 0 takes the entries (two ahead), publishes each to w2q / w2q7 and arms its ring stages (W2: all stages; others: the
// even stages); warp 7 arms the odd stages of W13 / shared down entries and, for W2, checks every byte of the h_q segment (and
// reloads it until all bytes are there); warps 1 and 6 issue the MMAs; warps 2..5 drain the accumulators and write the results.
// TMEM columns: 0..63 accumulators (runtime.cuh issue_job), 64..71 weight scales, 72..183 the 28 z_q scale chunks (phase switch),
// 184..231 the h_q scale chunks of the MAXSEG segments (3 each).
// Also the routing table of the step (build_table): slots = the experts used, in ascending id, each with its tokens and weights.
#pragma once
#include "runtime.cuh"
#include "moe_types.cuh"

namespace static_mk {

// one routing pair (token i / 16, k = i % 16) per call, i < 128: poll it (0xFF-prefilled), record it, count the expert's tokens
__device__ __forceinline__ void table_poll(G const &g, int i) {
  unsigned long long v;
  do { asm volatile("ld.relaxed.gpu.global.u64 %0, [%1];" : "=l"(v) : "l"(g.pairs64 + i) : "memory"); } while (v == ~0ull);
  int const e = (int)(v >> 32);
  sel_ss[i] = e; wsel_ss[i] = __uint_as_float((uint32_t)v);
  atomicAdd(&cntE_s[e], 1);
}

// after table_poll of all 128 pairs, 256 threads: slots = the experts with tokens, in ascending id (parallel prefix count);
// per slot its tokens and weights; s_ints[1] = number of slots
__device__ __forceinline__ void build_table() {
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __syncthreads();
  // warp w looks at experts [128 w, 128 w + 128): four 32-expert blocks b, one bit per lane
  int cnt_q[4]; uint32_t m_q[4];
#pragma unroll
  for (int q = 0; q < 4; q++) {
    int const b = warp * 4 + q;
    bool const live = (b < NE / 32) && cntE_s[b * 32 + lane] > 0;
    m_q[q] = __ballot_sync(0xffffffffu, live); cnt_q[q] = __popc(m_q[q]);
  }
  if (lane == 0) s_ints[16 + warp] = cnt_q[0] + cnt_q[1] + cnt_q[2] + cnt_q[3];
  __syncthreads();
  int wbase = 0;
  for (int w_ = 0; w_ < warp; w_++) wbase += s_ints[16 + w_];
  int n_live = 0;
  for (int w_ = 0; w_ < 8; w_++) n_live += s_ints[16 + w_];
  {
    int run = wbase;
#pragma unroll
    for (int q = 0; q < 4; q++) {
      int const b = warp * 4 + q;
      if (b < NE / 32) {
        if (cntE_s[b * 32 + lane] > 0) {
          int const slot = run + __popc(m_q[q] & ((1u << lane) - 1u));
          eid_s[slot] = b * 32 + lane; ntok_s[slot] = 0;
        }
        run += cnt_q[q];
      }
    }
  }
  if (threadIdx.x == 0) s_ints[1] = n_live;
  __syncthreads();
  if (threadIdx.x < T * 16) {   // one pair per thread: its slot by binary search over eid_s, its position by a shared atomic
    int const t = threadIdx.x >> 4, e = sel_ss[threadIdx.x];
    int lo = 0, hi = n_live - 1;
    while (lo < hi) { int const mid = (lo + hi) >> 1; if (eid_s[mid] < e) lo = mid + 1; else hi = mid; }
    int const pos_ = atomicAdd(&ntok_s[lo], 1);
    tok_s[lo * T + pos_] = t; w_s[lo * T + pos_] = wsel_ss[threadIdx.x];
  }
  __syncthreads();
}

// all 8 warps of the SM, until the queue is empty. gl / gi0 / gi1: ring stage counters of the loader / issuers; wt: accumulator
// tiles so far (the same on entry for every role)
__device__ __forceinline__ void run_expert_dynamic(G const &g, Maps const &maps, uint32_t base, uint32_t tb, int &gl, int &gi0, int &gi1, int &wt) {
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  // W2 items n2 = slots x OT2. The first ~4/5 of them (whole slots) go in W2_CHUNK-item chunks (nA chunks), the rest in 2-item
  // chunks, so that the last entries are small and the SMs finish close together.
  int const n_live = s_ints[1], n13 = n_live * MT13, n2 = n_live * OT2;
  int const nA = ((n2 * 4 / 5) / OT2) * (OT2 / W2_CHUNK), nchunks = nA + (n2 - nA * W2_CHUNK + 1) / 2;
  float *acc_s = reinterpret_cast<float *>(rt_sm() + OFF_ACC);
  auto entry_of = [&](int idx, QueueEntry &e) {   // entry number idx of the queue order above
    e.hq_buf = e.hq_use = 0;
    if (idx < n13) {
      e.kind = QE_W13; e.a = idx / MT13; e.b = idx % MT13;
    } else if (idx < n13 + nchunks) {
      int const c = idx - n13;
      int const i0 = (c < nA) ? c * W2_CHUNK : nA * W2_CHUNK + (c - nA) * 2, sz = (c < nA) ? W2_CHUNK : 2;
      e.kind = QE_W2; e.a = i0; e.b = (n2 - i0 < sz) ? n2 - i0 : sz;
    } else if (idx < n13 + nchunks + N_SDOWN) {
      int const q = idx - n13 - nchunks;
      e.kind = QE_SHARED_DOWN; e.a = q >> 1; e.b = q & 1;
    } else {
      e.kind = QE_END; e.a = e.b = 0;
    }
  };
  auto job_of = [&](QueueEntry const &e, TileJob &j) {   // the ring job of an entry
    j = TileJob{};
    if (e.kind == QE_W13) {
      int const ex = eid_s[e.a];
      j.kind = 1; j.natoms = KT_LAT; j.nst = (KT_LAT + 1) / 2; j.k0 = 0;
      j.wmap = &maps.w13; j.sfmap = &maps.w13sf; j.wmap2 = &maps.w13x2; j.sfmap2 = &maps.w13sfx2; j.amap = &maps.zq; j.amap2 = &maps.zq2;
      j.tile0 = j.chunk0 = ex * (MT13 * KT_LAT) + e.b * KT_LAT;
    } else if (e.kind == QE_W2) {
      int const slot = e.a / OT2, ex = eid_s[slot], ot0 = e.a - slot * OT2;
      j.kind = 2; j.natoms = e.b * KT2; j.nst = (j.natoms + 1) / 2;
      j.wmap = &maps.w2; j.sfmap = &maps.w2sf; j.wmap2 = &maps.w2x2; j.sfmap2 = &maps.w2sfx2;
      j.tile0 = j.chunk0 = ex * (OT2 * KT2) + ot0 * KT2;
    } else {   // QE_SHARED_DOWN
      j.kind = 0; j.nst = KT_SH / 2; j.row0 = e.a * 128; j.k0 = e.b * (KT_SH / 2); j.wmap = &maps.wsd; j.amap = &maps.hs;
    }
  };
  if (warp == 0) {
    if (lane == 0) {
      int qn = 0, cur_slot = -1, cur_use = 0, buf = -1, nbuf = 0, retired = 0;
      int buf_entry[MAXSEG];   // per h_q segment: the last entry that uses it
      bool hs_ok = false;
      for (int b = 0; b < MAXSEG; b++) buf_entry[b] = -1;
      // entry indices are taken TWO entries ahead: the atomic's round trip (> 1 us with 148 SMs) is never waited for
      int idx = (int)atomicAdd(g.cnt + C_W2NEXT, 1u), idx1 = (int)atomicAdd(g.cnt + C_W2NEXT, 1u);
      // h_q + scales of a W2 slot -> segment nbuf % MAXSEG (warp 7 checks the bytes, issuer 0 copies the scales to TMEM)
      auto load_hq = [&](int slot, int &b_out, int &use_out) {
        int const b = nbuf % MAXSEG;
        if (buf_entry[b] >= retired) {   // wait until the segment's previous entry is retired by the epilogue
          int const e_ = buf_entry[b];
          mbar_wait(rt.qempty0 + 8 * (e_ % W2QD), (e_ / W2QD) & 1);
          retired = e_ + 1;
        }
        uint32_t const hb = base + OFF_HQ2 + b * SEGB;
        mbar_expect(rt.hq0 + 8 * b, KT2 * (T * 128 + SF_CHUNK));
        tma3(&maps.hq3, rt.hq0 + 8 * b, hb, 0, slot * T, 0, EVICT_LAST);
        tma3(&maps.hsf3, rt.hq0 + 8 * b, hb + KT2 * 1024, 0, 0, slot * KT2, EVICT_LAST);
        b_out = b; use_out = nbuf / MAXSEG; nbuf++;
      };
      auto publish = [&](QueueEntry &e, int qn_) {   // prepare entry qn_ and write it to both shared-memory copies
        int const qpos = qn_ % W2QD;
        if (qn_ >= W2QD) {   // the entry that used this position is retired
          mbar_wait(rt.qempty0 + 8 * qpos, ((qn_ / W2QD) - 1) & 1);
          if (retired < qn_ - W2QD + 1) retired = qn_ - W2QD + 1;
        }
        if (e.kind == QE_W2) {
          int const slot = e.a / OT2;
          if (slot != cur_slot) { load_hq(slot, buf, cur_use); cur_slot = slot; }   // loaded early; warp 7 checks the bytes
          buf_entry[buf] = qn_; e.hq_buf = buf; e.hq_use = cur_use;
        } else if (e.kind == QE_SHARED_DOWN) {
          if (!hs_ok) { cnt_wait(g.cnt + C_HS, N_SACT); hs_ok = true; }   // h_s complete
          if (e.b == 0) {   // the 16 rows of this GPU's latent_up weight that match this S tile -> L2 (the tail uses them)
            unsigned char const *wu = g.w_up + ((size_t)g.rank * (H / TPMAX) + (size_t)e.a * 16) * LAT * 2;
            for (int q = 0; q < 4; q++) asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" :: "l"(wu + q * 28672), "r"(28672) : "memory");
          }
        }
        w2q[qpos][0] = e.kind; w2q[qpos][1] = e.a; w2q[qpos][2] = e.b; w2q[qpos][3] = e.hq_buf; w2q[qpos][4] = e.hq_use;
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        mbar_arrive(rt.qfull0 + 8 * qpos);
        if (qn_ >= W2QD) mbar_wait(rt.qempty7_0 + 8 * qpos, ((qn_ / W2QD) - 1) & 1);   // warp 7 has read the previous entry here
        w2q7[qpos][0] = e.kind; w2q7[qpos][1] = e.a; w2q7[qpos][2] = e.b; w2q7[qpos][3] = e.hq_buf; w2q7[qpos][4] = e.hq_use;
        mbar_arrive(rt.qfull7_0 + 8 * qpos);
      };
      QueueEntry e; entry_of(idx, e); publish(e, qn);
      while (e.kind != QE_END) {
        int const idx2 = (int)atomicAdd(g.cnt + C_W2NEXT, 1u);
        TileJob j; job_of(e, j);
        // W2: this warp arms ALL stages (two loaders arming W2 produced wrong results, 2026-09-10); others: the even stages
        auto arm = [&](int s0, int s1) { load_job(j, base, gl, e.kind == QE_W2 ? -1 : 0, s0, s1); };
        arm(0, 2);                                                  // the first two stages are in flight ...
        QueueEntry en; entry_of(idx1, en); publish(en, qn + 1);     // ... while the next entry is prepared and published
        arm(2, 1 << 20);
        e = en; qn++; idx = idx1; idx1 = idx2;
      }
    }
  } else if (warp == 1 || warp == 6) {
    if (lane == 0) {
      int const iss = (warp == 1) ? 0 : 1;
      int &gi = (warp == 1) ? gi0 : gi1;
      uint32_t const sfa = tb + 64 + 4 * iss;
      int wtt0 = wt, my_buf = -1, my_buse = -1;
      int vcnt[MAXSEG];   // per segment: how many times this issuer switched to it (the parity of hqv / sfb)
      for (int b = 0; b < MAXSEG; b++) vcnt[b] = 0;
      for (int qn = 0;; qn++) {
        int const qpos = qn % W2QD;
        mbar_wait(rt.qfull0 + 8 * qpos, (qn / W2QD) & 1);
        QueueEntry const e{w2q[qpos][0], w2q[qpos][1], w2q[qpos][2], w2q[qpos][3], w2q[qpos][4]};
        if (e.kind == QE_END) break;
        if (e.kind == QE_W2) {
          int const natoms = e.b * KT2, nst = (natoms + 1) / 2, buf = e.hq_buf;
          uint32_t const hb = base + OFF_HQ2 + buf * SEGB;
          if (buf != my_buf || e.hq_use != my_buse) {   // new h_q in this segment: issuer 0 copies its scales to TMEM once checked
            if (iss == 0) {
              mbar_wait(rt.hqv0 + 8 * buf, vcnt[buf] & 1); fence_after();
              for (int kt = 0; kt < KT2; kt++) cp_sf(tb + 184 + 12 * buf + 4 * kt, hb + KT2 * 1024 + kt * SF_CHUNK);
              tc_commit(rt.sfb0 + 8 * buf);
            }
            mbar_wait(rt.sfb0 + 8 * buf, vcnt[buf] & 1); fence_after();
            my_buf = buf; my_buse = e.hq_use; vcnt[buf]++;
          }
          // piece q = 2 st_ + iss belongs to item q / 3, K tile kt = q % 3. The issuer with kt 0 of an item also has kt 2, so it
          // starts the accumulation (first_mine = kt != 2) and the other one commits it (last_mine = kt != 0).
          uint64_t const dh0 = mkdesc(hb);
          uint32_t const sfb0c = tb + 184 + 12 * buf, acc0 = tb + 32 * iss;
          int item = iss / KT2, kt = iss % KT2;
          for (int st_ = 0; st_ < nst; st_++, gi++) {
            int const st = gi % SMAX;
            bool const has = 2 * st_ + iss < natoms;
            int const wtt = wtt0 + item, a = wtt & 3;
            bool const first_mine = kt != 2, last_mine = kt != 0;
            if (has && first_mine && wtt >= 4) mbar_wait(rt.acc_empty0 + 8 * a, ((wtt >> 2) - 1) & 1);
            mbar_wait(rt.full0 + 8 * st, (gi / SMAX) & 1); fence_after();
            if (has) {
              uint32_t const acc = acc0 + 8 * a, sfb = sfb0c + 4 * kt;
              uint64_t const dw = mkdesc(base + st * FSTAGE + iss * 16384), dh = dh0 + 64 * kt;
              cp_sf(sfa, base + OFF_WSF + (st * 2 + iss) * SF_CHUNK);
              mma_mx(acc, dw, dh, idesc_mx(0), first_mine ? 0u : 1u, sfa, sfb);
              mma_mx(acc, dw + 2, dh + 2, idesc_mx(1), 1u, sfa, sfb);
              mma_mx(acc, dw + 4, dh + 4, idesc_mx(2), 1u, sfa, sfb);
              mma_mx(acc, dw + 6, dh + 6, idesc_mx(3), 1u, sfa, sfb);
            }
            tc_commit(rt.empty0 + 8 * st);
            if (has && last_mine) tc_commit(rt.acc_full0 + 8 * a);
            kt += 2; if (kt >= KT2) { kt -= KT2; item++; }
          }
          wtt0 += e.b;
        } else {   // QE_W13, QE_SHARED_DOWN
          TileJob j; job_of(e, j);
          int const a = wtt0 & 3;
          if (wtt0 >= 4) mbar_wait(rt.acc_empty0 + 8 * a, ((wtt0 >> 2) - 1) & 1);
          issue_job(j, base, tb, iss, gi, a, tb + 72);
          wtt0++;
        }
      }
    }
  } else if (warp == 7) {   // second loader (odd stages of W13 / shared down) + the h_q segment check (all 32 lanes)
    int seen_buf = -1, seen_use = -1;
    int rph[MAXSEG];   // per segment: reloads so far (the parity of hqr)
    for (int b = 0; b < MAXSEG; b++) rph[b] = 0;
    for (int qn = 0;; qn++) {
      int const qpos = qn % W2QD;
      mbar_wait(rt.qfull7_0 + 8 * qpos, (qn / W2QD) & 1);
      QueueEntry const e{w2q7[qpos][0], w2q7[qpos][1], w2q7[qpos][2], w2q7[qpos][3], w2q7[qpos][4]};
      __syncwarp();
      if (lane == 0) mbar_arrive(rt.qempty7_0 + 8 * qpos);
      if (e.kind == QE_END) break;
      if (e.kind == QE_W2) {
        int const buf = e.hq_buf;
        if (buf != seen_buf || e.hq_use != seen_use) {   // new segment content: wait for warp 0's load, then check every byte
          seen_buf = buf; seen_use = e.hq_use;
          int const slot = e.a / OT2;
          uint32_t const hb = base + OFF_HQ2 + buf * SEGB;
          auto reload = [&]() {
            mbar_expect(rt.hqr0 + 8 * buf, KT2 * (T * 128 + SF_CHUNK));
            tma3(&maps.hq3, rt.hqr0 + 8 * buf, hb, 0, slot * T, 0, EVICT_LAST);
            tma3(&maps.hsf3, rt.hqr0 + 8 * buf, hb + KT2 * 1024, 0, 0, slot * KT2, EVICT_LAST);
          };
          mbar_wait(rt.hq0 + 8 * buf, e.hq_use & 1);   // warp 0's early load of this segment has landed
          // every W13 item of the slot has written its h_q (acquire); then reload the segment through the async proxy: the early
          // load may predate the bytes, and the h_q fill leaves no marker the byte check below could see
          if (lane == 0) {
            cnt_wait(g.cnt + C_HQ + slot, MT13);
            asm volatile("fence.proxy.async.global;" ::: "memory");
            reload();
          }
          __syncwarp(); mbar_wait(rt.hqr0 + 8 * buf, rph[buf] & 1); rph[buf]++;
          for (;;) {   // h_q bytes: no 0xFF byte (0xFF-prefilled); scale bytes: none zero (all >= 19)
            bool ok = true;
            for (int i = lane; i < KT2 * 64; i += 32) {
              uint4 v;
              asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(hb + i * 16));
              ok = ok && !(has_ff(v.x) | has_ff(v.y) | has_ff(v.z) | has_ff(v.w));
            }
            if (lane < KT2 * T) {
              uint32_t w;
              asm volatile("ld.shared.u32 %0, [%1];" : "=r"(w) : "r"(hb + KT2 * 1024 + (lane / T) * SF_CHUNK + (lane % T) * 16));
              ok = ok && ((w & 0xFFu) && (w & 0xFF00u) && (w & 0xFF0000u) && (w & 0xFF000000u));
            }
            if (__all_sync(0xffffffffu, ok)) break;
            if (lane == 0) { __nanosleep(300); reload(); }
            __syncwarp(); mbar_wait(rt.hqr0 + 8 * buf, rph[buf] & 1); rph[buf]++;
          }
          __syncwarp();
          if (lane == 0) mbar_arrive(rt.hqv0 + 8 * buf);
        }
        gl += (e.b * KT2 + 1) / 2;   // W2 stages are all armed by warp 0; keep this counter in step
      } else {
        TileJob j; job_of(e, j);
        if (lane == 0) load_job(j, base, gl, 1); else gl += j.nst;
      }
    }
  } else if (warp >= 2 && warp <= 5) {
    int wtt0 = wt;
    for (int qn = 0;; qn++) {
      int const qpos = qn % W2QD;
      mbar_wait(rt.qfull0 + 8 * qpos, (qn / W2QD) & 1);
      QueueEntry const e{w2q[qpos][0], w2q[qpos][1], w2q[qpos][2], w2q[qpos][3], w2q[qpos][4]};
      if (e.kind == QE_END) break;
      if (e.kind == QE_W2) {   // R[t][128 ot + row] += weight(t) * result, for the slot's tokens
        for (int item = 0; item < e.b; item++) {
          int const wtt = wtt0 + item, a = wtt & 3, par = (wtt >> 2) & 1, w = e.a + item, slot = w / OT2, ot = w - slot * OT2, nt = ntok_s[slot];
          float v[8];
          int const row = drain_acc(tb, a, par, v);
          for (int i = 0; i < nt; i++) {
            int const t = tok_s[slot * T + i];
            red_add(g.Racc + t * LAT + ot * 128 + row, w_s[slot * T + i] * v[t]);
          }
        }
        wtt0 += e.b;
      } else if (e.kind == QE_W13) {   // rows 0..63 gate, 64..127 up of features [64 m, 64 m + 64) -> h_q of the slot
        int const slot = e.a, mt = e.b, a = wtt0 & 3, par = (wtt0 >> 2) & 1;
        float v[8];
        int const row = drain_acc(tb, a, par, v);
        for (int t = 0; t < T; t++) acc_s[row * 8 + t] = v[t];
        asm volatile("bar.sync 1, 128;" ::: "memory");
        if (row < 64) {   // feature f = 64 mt + row: gate acc[row], up acc[64 + row]; one 32-feature scale group per warp
          int const f = mt * 64 + row, kt2 = f >> 7, kb = (f >> 5) & 3;
          for (int t = 0; t < T; t++) {
            float const h = situ(acc_s[row * 8 + t], acc_s[(64 + row) * 8 + t]);
            float amax = fabsf(h);
            for (int d = 16; d > 0; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
            amax = fmaxf(amax, 1.0e-30f);
            float ex = ceilf(log2f(amax * (1.0f / 448.0f)));
            ex = fminf(fmaxf(ex, -127.f), 127.f);
            g.hq[((size_t)slot * T + t) * IR + f] = cvt_e4m3(h * exp2f(-ex));
            if (lane == 0) g.hsf[((size_t)slot * KT2 + kt2) * SF_CHUNK + t * 16 + kb] = (uint8_t)(int)(ex + 127.f);
          }
        }
        __threadfence();   // this thread's h_q / scale bytes before the item is counted
        asm volatile("bar.sync 1, 128;" ::: "memory");
        if (threadIdx.x == 64) cnt_add(g.cnt + C_HQ + slot, 1);   // release; warp 7 acquires MT13 of these per slot
        wtt0++;
      } else {   // QE_SHARED_DOWN: S rows [128 a, 128 a + 128); the two K halves add into the zeroed Sout
        int const a = wtt0 & 3, par = (wtt0 >> 2) & 1;
        float v[8];
        int const row = drain_acc(tb, a, par, v);
        for (int t = 0; t < T; t++) red_add(g.Sout + t * H + e.a * 128 + row, v[t]);
        wtt0++;
      }
      mbar_arrive(rt.qempty0 + 8 * qpos);   // 128 arrivals: entry retired
    }
  }
}

}  // namespace static_mk
