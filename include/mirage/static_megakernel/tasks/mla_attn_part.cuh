// tasks/mla_attn_part.cuh -- Kimi K3 MLA: the attention of a group of HG heads
// over one KV split, and the splits' combine (SGLang: trtllm-gen MLA decode,
// split over CTAs, and its combine). Task (g, s) of grid (heads / HG, NS) =
// heads [HG g, + HG) of this GPU, the KV rows of split s:
//   queries  qi = hh T + t (head HG g + hh, token t): [q_abs (KV) | q_pe
//   (ROPE)] (mla_absorb's output [T][heads][KV + ROPE]
//            bf16, 0xFF-prefilled), HG T rows in shared memory
//   rows     the request's n = seq_len_k KV rows: rows j < n - T from the
//   framework's paged KV cache (page table pt: row j at
//            cache row pt[j / PS] * PS + j % PS), the last T = this step's
//            (mla_norm's kv_new, 0xFF-prefilled); 64-row chunks, nch of them,
//            over nne splits (NS when nch >= 3 NS, else min(NS - LATE, nch):
//            LATE = params[7], the last splits' SMs start late): split s < nne
//            takes chunks ceil(s nch / nne) .. ceil((s + 1) nch / nne) - 1 (the
//            last one has this step's rows), the others none. Only contexts
//            over thr rows (shorter ones: mla_attention's per-head tasks; thr
//            0: every context here). One split with rows (at most CH rows): O /
//            l straight to attn in bf16, no partial and no combine
//   partial  per query: m = max_j score, l = sum_j exp(score - m), a = sum_j
//   exp(score - m) row_j[:KV] / l,
//            score_j = scale (query . row_j) for j <= n - T + t (token t sees
//            the rows before it and the step's up to itself)
//   attn     = sum_s w_s a_s, w_s = l_s exp(m_s - M) / sum_s' l_s' exp(m_s' -
//   M) (M = max_s m_s), rounded to bf16 (SGLang's
//            attention writes bf16) -> the node's output [T][heads][KV] bf16
//            (0xFF before each launch: mla_attention's split 0 polls it per 16
//            B). Every split writes its partial (a [groups][NS][HG T][KV] fp32,
//            then m / l [groups][NS][HG T][2] fp32 with release, 0xFF before
//            each launch), waits for the group's nne partials (acquire) and
//            combines its 1 / nne share of the group's [HG T][KV] outputs (one
//            contiguous range of each partial: one bulk copy each). (a in bf16
//            doubled the error at 8000 rows, 10-11, for no time: the exchange's
//            cost is its latency, ~3.7 us SM to SM.)
// The chunks: 5 TMA boxes of the cache (map kv_map: [rows][KV + ROPE] bf16 as
// 64-element blocks; a box = 64 rows x 2 blocks, 128-B swizzle: ldmatrix
// without bank conflicts; the 5th box's second block past the row, zero-filled)
// completing on an mbarrier per buffer, issued by one thread; the page numbers
// read once at the start; two buffers (chunk ci + 1 copied while ci is
// computed); the rows >= n - T written over after the copy (this step's rows,
// zeros past n). The step's rows and the queries are polled together at the
// start (cp.async into shared memory, checked there: no registers held, the PV
// accumulators need them). Scores S^T [CH][HG T] by tcgen05 MMAs into TMEM (A =
// the chunk's rows, B = the queries, both K-major 128-B swizzled in shared
// memory); an online softmax, 16 threads per query; O^T [KV][HG T] += rows^T .
// P^T by mma.sync (M = 16 KV dims via ldmatrix.trans, N = 8 queries, K = the
// chunk's rows; P in bf16): warp w KV dims [64 w, + 64) x the HG query blocks.
#pragma once
#include "../core.cuh"
#include "gemm_tile.cuh"     // ldsm_x4, ldsm_x2, mma_16816, valid16h
#include "mla_attention.cuh" // ldsm_x4_t, MlaKv
#include "mla_common.cuh"

namespace static_mk {

constexpr int MLA_CH = 64,
              MLA_MAXC = 64; // rows per chunk; chunks per split whose page
                             // numbers are held in shared memory

__device__ __forceinline__ uint32_t pack_bf16x2(float lo, float hi) {
  __nv_bfloat162 const v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t const *>(&v);
}

// The splits' combine, run by every split s < nne of group g after its partial
// a (part_o) is stored: its m / l (from ml) with release, the group's nne m / l
// (acquire), the weights, and this split's 1 / nne share of the group's
// [NQ][KV] outputs -> attn bf16. Shared memory: rows0 (the row buffers, free
// now) and wt [NS][NQ] in Sp; mbarrier mb (count 1) at phase mph.
template <int HEADS, int HG, int KV, int NS>
__device__ __forceinline__ void mla_part_combine(char *const rows0,
                                                 float *const Sp,
                                                 float const *ml,
                                                 float const *part_o,
                                                 float *part_ml,
                                                 __nv_bfloat16 *attn,
                                                 int g,
                                                 int s,
                                                 int nne,
                                                 uint32_t mb,
                                                 int mph) {
  constexpr int NQ = HG * T, U = NQ * KV / 4, LPQ = 256 / NQ,
                MAXJ = (NS + LPQ - 1) / LPQ; // weight lanes per query
  int const tid = threadIdx.x;
  auto out_row = [&](int qi) {
    return attn + ((size_t)(qi % T) * HEADS + HG * g + qi / T) * KV;
  };
  // rows0: slots [nne][nu] float4 | mls [NS][NQ][2] fp32
  float4 *const slot = reinterpret_cast<float4 *>(rows0);
  float *const mls = reinterpret_cast<float *>(rows0 + (size_t)(U + NS) * 16);
  float(*const wt)[NQ] = reinterpret_cast<float(*)[NQ]>(Sp);
  if (tid < NQ) {
    float *const pml = part_ml + ((size_t)g * NS + s) * NQ * 2;
    asm volatile(
        "st.release.gpu.global.v2.f32 [%0], {%1, %2};" ::"l"(pml + 2 * tid),
        "f"(ml[tid * 4]),
        "f"(ml[tid * 4 + 1])
        : "memory");
  }
  // the group's nne partials' m / l (acquire) -> mls; per (split, query) its
  // weight w = l_s exp(m_s - M) / sum_s' l_s' exp(m_s' - M) -> wt
  for (int i = tid; i < nne * NQ; i += 256) {
    int const s2 = i / NQ, qi = i % NQ;
    float const *p = part_ml + (((size_t)g * NS + s2) * NQ + qi) * 2;
    float2 v;
    do {
      asm volatile("ld.acquire.gpu.global.v2.f32 {%0, %1}, [%2];"
                   : "=f"(v.x), "=f"(v.y)
                   : "l"(p)
                   : "memory");
    } while (!validf(v.x) || !validf(v.y));
    mls[s2 * NQ * 2 + qi * 2] = v.x;
    mls[s2 * NQ * 2 + qi * 2 + 1] = v.y;
  }
  __syncthreads();
  // this split's share: float4 units [s U / nne, (s + 1) U / nne) of the
  // group's U = NQ KV / 4 (unit u: query u / (KV / 4), dims 4 (u % (KV / 4))
  // ..): in each partial one contiguous range, copied by one bulk copy (warp
  // 0's elected lane, nne of them, on mbarrier mb) into the row buffers, slot
  // s2 = partial s2's range; the weights computed while they fly. (10-13: 16-B
  // cp.async per thread and partial: their issue 1.2 us and a split-quarter
  // sum 1.7 us at 8000 rows; the weights by 16 threads looping over the
  // splits: 4.3 us; the copies issued by thread 0 alone, each through an elect
  // / broadcast loop: 2 us for the 21)
  int const u0 = s * U / nne, u1 = (s + 1) * U / nne, nu = u1 - u0;
  float const *pg = part_o + (size_t)g * NS * NQ * KV;
  if ((tid >> 5) == 0) {
    asm volatile("fence.proxy.async.global;" ::
                     : "memory"); // the partials (generic writes, seen through
                                  // the acquires)
    asm volatile(
        "fence.proxy.async.shared::cta;" ::
            : "memory"); // the row buffers' generic reads -> the copies
    if (issuer_lane<true>()) {
      mbar_expect(mb, (uint32_t)(nne * nu * 16));
      for (int s2 = 0; s2 < nne; s2++) {
        asm volatile(
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
            "[%0], [%1], %2, [%3];" ::"r"(su32(slot + (size_t)s2 * nu)),
            "l"(pg + (size_t)s2 * NQ * KV + (size_t)u0 * 4),
            "r"((uint32_t)(nu * 16)),
            "r"(mb)
            : "memory");
      }
    }
    __syncwarp();
  }
  { // the weights: query qi = tid / LPQ, its lane j = tid % LPQ splits j, j +
    // LPQ, ..; max and sum over the LPQ lanes by shuffles
    int const qi = tid / LPQ, j = tid % LPQ;
    float m[MAXJ], l[MAXJ];
    float M = -INFINITY;
#pragma unroll
    for (int k = 0; k < MAXJ; k++) {
      int const s2 = j + k * LPQ;
      m[k] = s2 < nne ? mls[s2 * NQ * 2 + qi * 2] : -INFINITY;
      l[k] = s2 < nne ? mls[s2 * NQ * 2 + qi * 2 + 1] : 0.f;
      M = fmaxf(M, m[k]);
    }
#pragma unroll
    for (int dd = 1; dd < LPQ; dd <<= 1) {
      M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, dd));
    }
    float Lt = 0.f;
#pragma unroll
    for (int k = 0; k < MAXJ; k++) {
      l[k] = l[k] == 0.f ? 0.f : l[k] * __expf(m[k] - M);
      Lt += l[k];
    }
#pragma unroll
    for (int dd = 1; dd < LPQ; dd <<= 1) {
      Lt += __shfl_xor_sync(0xffffffffu, Lt, dd);
    }
    float const rl = 1.f / Lt;
#pragma unroll
    for (int k = 0; k < MAXJ; k++) {
      if (j + k * LPQ < nne) {
        wt[j + k * LPQ][qi] = l[k] * rl;
      }
    }
  }
  __syncthreads();
  mbar_wait(mb, mph);
  for (int u = u0 + tid; u < u1; u += 256) {
    int const qi = u / (KV / 4), d = (u % (KV / 4)) * 4;
    float4 o = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll 4
    for (int s2 = 0; s2 < nne; s2++) {
      float const w = wt[s2][qi];
      float4 const v = slot[s2 * nu + (u - u0)];
      o.x += w * v.x;
      o.y += w * v.y;
      o.z += w * v.z;
      o.w += w * v.w;
    }
    *reinterpret_cast<uint2 *>(out_row(qi) + d) =
        make_uint2(pack_bf16x2(o.x, o.y), pack_bf16x2(o.z, o.w));
  }
}

// Returns whether it used the shared memory. (Registers: with 4 heads a task,
// 64 accumulators per thread, and mla_attention's per-head tasks in the same
// kernel, ptxas spilled the kernel's per-SM state, reloaded in the GEMMs' TMA
// loops: every front task 1.3 - 2.3 us later, 10-11; as a call it spilled more.
// 2 heads: 32 accumulators.)
template <int HEADS, int HG, int KV, int ROPE, int PS, int NS, int LATE = 0>
__device__ __forceinline__ bool
    mla_attn_part_task(char *const sm,
                       uint32_t const tb,
                       unsigned long long *const stage_stamps,
                       CUtensorMap const *kv_map,
                       __nv_bfloat16 const *q,
                       MlaKv const kv,
                       float scale,
                       float *part_o,
                       float *part_ml,
                       __nv_bfloat16 *attn,
                       int thr,
                       int g,
                       int s) {
  static_assert(KV == 512 && ROPE == 64 && T == 8,
                "mla_attn_part: the K3 shapes, 8 tokens");
  constexpr int D = KV + ROPE, RB = D * 2, RS = RB + 16, CH = MLA_CH,
                NQ = HG * T, QB = NQ / 8; // QB query blocks of 8
  constexpr int NBOX = (D / 64 + 1) / 2, BOX = 2 * CH * 128,
                CHB = NBOX * BOX; // TMA boxes per chunk
  static_assert(NQ == 16 && PS % CH == 0 && NS <= 48 && CH == 64,
                "mla_attn_part: 2 heads x 8 tokens (the scores' TMEM load: 16 "
                "columns), chunks within a page");
  static_assert((NQ * KV / 4 + NS) * 16 + NS * NQ * 8 <= 2 * CHB &&
                    NS * NQ <= NQ * CH,
                "mla_attn_part: the combine's shared memory");
  int const tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  // the splits used: all NS when each gets at least 3 chunks, else not the LATE
  // last ones (on SMs that start them late: their start is hidden only behind
  // several chunks; 10-13: a 21st split on SMs 40..45 cost 4.2 us at 2000 rows,
  // saved 1.5 at 8000)
  int const n = *kv.seq_k, L0 = n - T, nch = (n + CH - 1) / CH,
            nne = nch >= 3 * NS ? NS : min(NS - LATE, nch);
  int const cs = (s * nch + nne - 1) / nne,
            ce = s < nne ? ((s + 1) * nch + nne - 1) / nne : cs;
  if (n <= thr ||
      cs >= ce) { // a short context (mla_attention's per-head tasks) or an
                  // empty split (the combine reads splits
    // 0 .. nne - 1): nothing
    return false;
  }
  // layout: rows [2][NBOX][2 blocks][CH][128 B] | Qs [D / 64][NQ][128 B] (the
  // MMAs' B: K-major, 128-B swizzled) | kvn [T][RS]
  //         (this step's rows) | S [NQ][CH] fp32 | P [NQ][CH] bf16 (in a
  //         [NQ][CH] fp32 region) | ml [NQ][4] fp32 (m, l, the chunk's rescale)
  //         | pages [MAXC] int | 3 mbarriers (the 2 buffers' copies, the
  //         scores' MMAs)
  char *const rows0 = sm;
  char *const qs = rows0 + 2 * CHB;
  char *const kvn = qs + NQ * RB;
  float *const S = reinterpret_cast<float *>(kvn + T * RS);
  float *const Sp = S + NQ * CH;
  __nv_bfloat16 *const P = reinterpret_cast<__nv_bfloat16 *>(Sp);
  float *const ml = Sp + NQ * CH;
  int *const pages = reinterpret_cast<int *>(ml + NQ * 4);
  uint32_t const mb =
      su32(pages +
           MLA_MAXC); // 8-B aligned: every region above is a multiple of 16 B
  uint32_t const qbase = su32(qs);
  // a chunk row's 16-B chunk c (block c / 8) in its buffer: box block / 2,
  // block block % 2, 128-B swizzled by the row
  auto rofs = [](int r, int c) {
    return (c >> 4) * BOX + ((c >> 3) & 1) * (CH * 128) + r * 128 +
           (((c & 7) ^ (r & 7)) << 4);
  };
  __syncthreads(); // the SM's previous task is done with the shared memory
  if (tid == 0) {
    mbar_init(mb, 1);
    mbar_init(mb + 8, 1);
    mbar_init(mb + 16, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  for (int i = tid; i < min(ce - cs, MLA_MAXC); i += 256) {
    int const j0 = (cs + i) * CH;
    pages[i] = j0 < L0 ? kv.pt[j0 / PS] : 0;
  }
  if (tid < NQ) {
    ml[tid * 4 + 0] = -INFINITY;
    ml[tid * 4 + 1] = 0.f;
  }
  __syncthreads();
  // chunk ci into buffer b (if it has cache rows, j < L0): thread it (0, or 128
  // in the loop), the NBOX boxes of its 64 cache rows
  auto issue = [&](int ci, int b, int it) {
    if (tid == it) {
      int const j0 = ci * CH;
      asm volatile("fence.proxy.async.shared::cta;" ::
                       : "memory"); // the buffer's generic reads / writes
                                    // before the copies
      if (j0 < L0) {
        int const page = ci - cs < MLA_MAXC ? pages[ci - cs] : kv.pt[j0 / PS];
        mbar_expect(mb + 8 * b, CHB);
        uint32_t const dst = su32(rows0 + b * CHB);
#pragma unroll
        for (int x = 0; x < NBOX; x++) {
          tma3(kv_map,
               mb + 8 * b,
               dst + x * BOX,
               0,
               page * PS + j0 % PS,
               2 * x,
               EVICT_FIRST); // (10-14: EVICT_LAST kept the KV rows in L2 after
                             // the layer; same one-layer span, less MoE overlap
                             // lost)
        }
      } else {
        mbar_arrive(mb + 8 * b);
      }
    }
  };
  issue(cs, 0, 0);
  // O^T [KV][NQ]: warp w KV dims [64 w, + 64) (4 blocks of 16) x QB query
  // blocks, per thread 4 x QB x 4 Pass ci = cs - 1 is dry: the chunk code run
  // once on buffer 1 (not yet copied into) with every score masked, while the
  // queries are not here yet, its results dropped: the code is then cached for
  // the chunks (10-11: cold code doubled a chunk).
  float acc[4][QB][4] = {};
  int sph =
      0; // the scores' mbarrier phase (one MMA batch per pass, the dry one too)
#pragma unroll 1
  for (int ci = cs - 1; ci < ce; ci++) {
    bool const dry = ci < cs;
    int const b = dry ? 1 : (ci - cs) & 1, ph = ((ci - cs) >> 1) & 1,
              j0 = dry ? n + CH : ci * CH;
    char *const rows = rows0 + b * CHB;
    uint32_t const rbase = su32(rows);
    if (ci == cs) {
      // the HG heads' queries -> Qs (row qi = hh T + t) and, if this split has
      // them, this step's rows -> kvn: polled together
      {
        constexpr int NV = NQ * (D / 8), PER = (NV + 255) / 256,
                      NK = T * (D / 8), PK = (NK + 255) / 256;
        static_assert(PER + PK <= 32, "mla_attn_part: the pending mask");
        bool const step_rows = ce * CH > L0;
        uint32_t pend = 0;
#pragma unroll
        for (int i = 0; i < PER; i++) {
          pend |= (tid + 256 * i < NV ? 1u : 0u) << i;
        }
#pragma unroll
        for (int i = 0; i < PK; i++) {
          pend |= (step_rows && tid + 256 * i < NK ? 1u : 0u) << (PER + i);
        }
        poll16_smem<PER + PK>(
            pend,
            [&](int i) -> void const * {
              if (i < PER) {
                int const v = tid + 256 * i, qi = v / (D / 8), c = v % (D / 8);
                return q + ((size_t)(qi % T) * HEADS + HG * g + qi / T) * D +
                       c * 8;
              }
              int const v = tid + 256 * (i - PER), t = v / (D / 8),
                        c = v % (D / 8);
              return kv.kv_new + (size_t)t * D + c * 8;
            },
            [&](int i) -> char * {
              if (i < PER) {
                int const v = tid + 256 * i, qi = v / (D / 8), c = v % (D / 8);
                return qs + (c >> 3) * (NQ * 128) + qi * 128 +
                       (((c ^ qi) & 7) << 4);
              }
              int const v = tid + 256 * (i - PER), t = v / (D / 8),
                        c = v % (D / 8);
              return kvn + t * RS + c * 16;
            });
        asm volatile("fence.proxy.async.shared::cta;" ::
                         : "memory"); // the queries (generic writes) -> the
                                      // MMAs (async proxy)
      }
      // the dry pass's state dropped (after the poll: the accumulators not live
      // across it)
#pragma unroll
      for (int mb_ = 0; mb_ < 4; mb_++) {
#pragma unroll
        for (int qb = 0; qb < QB; qb++) {
#pragma unroll
          for (int e = 0; e < 4; e++) {
            acc[mb_][qb][e] = 0.f;
          }
        }
      }
      if (tid < NQ) {
        ml[tid * 4 + 0] = -INFINITY;
        ml[tid * 4 + 1] = 0.f;
      }
#if defined(STATIC_TIMING_BUILD)
      __syncthreads();
      if (tid == 0 && stage_stamps) { // stage 0: the queries landed
        stage_stamps[0] = gtime();
      }
#endif
    }
    __syncthreads(); // the other buffer's chunk ci - 1 no longer read; kvn
                     // landed (every thread's poll)
    // the next chunk's copy into the other buffer, free from here (10-13:
    // issued after the scores, it overlapped only the softmax and P.V; a chunk
    // is 80 KB, ~1.1 us at an SM's ~65-70 GB/s), by thread 128 (thread 0's
    // issue cost the chain 0.25 us a chunk, 10-11 bench)
    if (!dry && ci + 1 < ce) {
      issue(ci + 1, b ^ 1, 128);
    }
    if (!dry) {
      mbar_wait(mb + 8 * b, ph);
    }
    bool const fill =
        !dry && j0 + CH > L0; // rows >= L0 in this chunk, over the copy: this
                              // step's (kvn), then zeros past n
    if (fill) {
      int const r0 = max(L0 - j0, 0);
      for (int i = tid; i < (CH - r0) * (RB / 16); i += 256) {
        int const r = r0 + i / (RB / 16), c = i % (RB / 16), j = j0 + r;
        *reinterpret_cast<uint4 *>(rows + rofs(r, c)) =
            j < n
                ? *reinterpret_cast<uint4 const *>(kvn + (j - L0) * RS + c * 16)
                : make_uint4(0u, 0u, 0u, 0u);
      }
      asm volatile("fence.proxy.async.shared::cta;" ::
                       : "memory"); // the rows written -> the MMAs
      __syncthreads();
    }
    // scores S^T [CH][NQ] = rows . Q^T on the tensor cores: tcgen05 MMAs (M =
    // CH rows, N = NQ queries, K = 16 each, D / 16 of them, issued by one
    // elected lane of warp 0) into TMEM columns tb + 256 .. + NQ (the GEMMs'
    // accumulators and scales use columns < 256); M = 64: row r in lane 32 (r /
    // 16) + r % 16, read by warps 0..3 (a warp reaches only its 32-lane slice).
    // (10-13: with mma.sync, the warps re-reading the queries' fragments,
    // shared-memory bound: 1.27 us of a chunk's 2.6, + 0.38 the k-halves' sum.)
    {
      constexpr uint32_t IDS = (1u << 4) | (1u << 7) | (1u << 10) |
                               ((uint32_t)(NQ / 8) << 17) |
                               ((uint32_t)(CH / 16) << 24);
      uint32_t const ts = tb + 256;
      if (warp == 0) { // the whole warp runs the issue with the same values
                       // (uniform registers), one elected lane issues
        // (10-13: issued by thread 0 alone, every MMA went through an elect /
        // broadcast loop: 1.25 us for the 36)
        fence_after();
        if (issuer_lane<true>()) {
          // descriptors stepped in place (start address in 16-B units: K 16 =
          // 2, the next block CH 128 B, the next box BOX)
          uint64_t da = mkdesc(rbase), dq = mkdesc(qbase);
#pragma unroll 1
          for (int kb = 0; kb < D / 64; kb++) {
#pragma unroll
            for (int k = 0; k < 4; k++) {
              asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, "
                           "0;\ntcgen05.mma.cta_group::1.kind::f16 [%0], %1, "
                           "%2, %3, p;\n}" ::"r"(ts),
                           "l"(da + 2 * k),
                           "l"(dq + 2 * k),
                           "r"(IDS),
                           "r"((kb | k) ? 1u : 0u));
            }
            da += (kb & 1) ? (uint64_t)((BOX - CH * 128) >> 4)
                           : (uint64_t)((CH * 128) >> 4);
            dq += (uint64_t)((NQ * 128) >> 4);
          }
          tc_commit(mb + 16);
        }
        __syncwarp();
      }
      mbar_wait(mb + 16, sph);
      sph ^= 1;
      fence_after();
      if (warp < 4) {
        float v[16];
        asm volatile(
            "tcgen05.ld.sync.aligned.32x32b.x16.b32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15}, [%16];"
            : "=f"(v[0]),
              "=f"(v[1]),
              "=f"(v[2]),
              "=f"(v[3]),
              "=f"(v[4]),
              "=f"(v[5]),
              "=f"(v[6]),
              "=f"(v[7]),
              "=f"(v[8]),
              "=f"(v[9]),
              "=f"(v[10]),
              "=f"(v[11]),
              "=f"(v[12]),
              "=f"(v[13]),
              "=f"(v[14]),
              "=f"(v[15])
            : "r"(ts + ((uint32_t)(warp * 32) << 16)));
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        if (lane < 16) {
          int const r = warp * 16 + lane, j = j0 + r;
#pragma unroll
          for (int qi = 0; qi < NQ; qi++) {
            S[qi * CH + r] = j <= L0 + qi % T ? v[qi] * scale : -INFINITY;
          }
        }
      }
      fence_before();
    }
    __syncthreads();
    { // online softmax: query qi = tid / TPQ, its columns [VPT (tid % TPQ), +
      // VPT)
      constexpr int TPQ = 256 / NQ, VPT = CH / TPQ;
      int const qi = tid / TPQ, sub = tid % TPQ;
      float x[VPT];
#pragma unroll
      for (int k = 0; k < VPT; k += 4) {
        *reinterpret_cast<float4 *>(x + k) =
            *reinterpret_cast<float4 const *>(S + qi * CH + sub * VPT + k);
      }
      float mx = x[0];
#pragma unroll
      for (int k = 1; k < VPT; k++) {
        mx = fmaxf(mx, x[k]);
      }
#pragma unroll
      for (int d = 1; d < TPQ; d <<= 1) {
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, d));
      }
      float const m_old = ml[qi * 4], m_new = fmaxf(m_old, mx);
      float sum = 0.f;
      __align__(16) __nv_bfloat16 pb[VPT];
#pragma unroll
      for (int k = 0; k < VPT; k++) {
        float const e = x[k] == -INFINITY ? 0.f : __expf(x[k] - m_new);
        pb[k] = __float2bfloat16(e);
        sum += e; // l over the unrounded exps (SGLang's attention: bf16 P for
                  // P.V, fp32 l)
      }
      { // P row qi: 8 chunks of 16 B, chunk c at (c ^ qi) & 7
        int const col = sub * VPT, c = col >> 3;
        __nv_bfloat16 *const dst =
            P + qi * CH + (((c ^ qi) & 7) << 3) + (col & 7);
        if constexpr (VPT == 8) {
          *reinterpret_cast<uint4 *>(dst) =
              *reinterpret_cast<uint4 const *>(pb);
        } else {
          *reinterpret_cast<uint2 *>(dst) =
              *reinterpret_cast<uint2 const *>(pb);
        }
      }
#pragma unroll
      for (int d = 1; d < TPQ; d <<= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, d);
      }
      __syncwarp();
      if (sub == 0) {
        float const resc = m_old == -INFINITY ? 0.f : __expf(m_old - m_new);
        ml[qi * 4] = m_new;
        ml[qi * 4 + 1] = ml[qi * 4 + 1] * resc + sum;
        ml[qi * 4 + 2] = resc;
      }
    }
    __syncthreads();
    { // O^T [KV][NQ] = O^T * resc + rows^T [KV][CH] . P^T
      int const t0 = 2 * (lane & 3);
#pragma unroll
      for (int qb = 0; qb < QB; qb++) {
        float const ra = ml[(qb * 8 + t0) * 4 + 2],
                    rb2 = ml[(qb * 8 + t0 + 1) * 4 + 2];
#pragma unroll
        for (int mb_ = 0; mb_ < 4; mb_++) {
          acc[mb_][qb][0] *= ra;
          acc[mb_][qb][1] *= rb2;
          acc[mb_][qb][2] *= ra;
          acc[mb_][qb][3] *= rb2;
        }
      }
      uint32_t const pbase = su32(P);
#pragma unroll
      for (int ks = 0; ks < CH / 16; ks++) {
        uint32_t b[QB][2]; // P^T fragments of the QB query blocks (8 registers;
                           // the A operand one block at a time: the
                           // accumulators' 64 registers leave little room)
#pragma unroll
        for (int qb = 0; qb < QB; qb++) {
          int const qi = qb * 8 + (lane & 7), c = 2 * ks + ((lane >> 3) & 1);
          ldsm_x2(pbase + qi * (CH * 2) + (((c ^ qi) & 7) << 4),
                  b[qb][0],
                  b[qb][1]);
        }
#pragma unroll
        for (int mb_ = 0; mb_ < 4; mb_++) {
          int const i = lane >> 3, r = ks * 16 + (i >> 1) * 8 + (lane & 7);
          int const c = (warp * 64 + mb_ * 16) / 8 + (i & 1);
          uint32_t a[4];
          ldsm_x4_t(rbase + rofs(r, c), a);
#pragma unroll
          for (int qb = 0; qb < QB; qb++) {
            mma_16816(acc[mb_][qb], a, b[qb][0], b[qb][1]);
          }
        }
      }
    }
  }
#if defined(STATIC_TIMING_BUILD)
  if (tid == 0 && stage_stamps) { // stage 1: the chunks done (warp 0)
    stage_stamps[1] = gtime();
  }
#endif
  __syncthreads(); // every warp's last PV done: the row buffers are free (the
                   // bf16 staging below)
  // the output row of query qi: head HG g + qi / T, token qi % T
  auto out_row = [&](int qi) {
    return attn + ((size_t)(qi % T) * HEADS + HG * g + qi / T) * KV;
  };
  int const t0 = 2 * (lane & 3), g8 = lane >> 2;
  // (a query that sees no row of this split, l = 0: token t < T - 1 when the
  // step's rows straddle two chunks and this split has only the last; its a
  // written 0, not 0 / 0 (its weight is 0, and 0 x NaN is NaN: 10-11, 700
  // rows))
  auto rcp = [](float l) { return l > 0.f ? 1.f / l : 0.f; };
  if (nne == 1) { // one split has the group's every chunk (a context of at most
                  // CH rows): O / l straight to attn in bf16, no
                  // partial and no combine
#pragma unroll
    for (int qb = 0; qb < QB; qb++) {
      float const ra = rcp(ml[(qb * 8 + t0) * 4 + 1]),
                  rb2 = rcp(ml[(qb * 8 + t0 + 1) * 4 + 1]);
#pragma unroll
      for (int mb_ = 0; mb_ < 4; mb_++) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
          out_row(qb * 8 + t0 +
                  (e & 1))[warp * 64 + mb_ * 16 + g8 + (e >> 1) * 8] =
              __float2bfloat16(acc[mb_][qb][e] * ((e & 1) ? rb2 : ra));
        }
      }
    }
    __syncthreads();
    if (tid == 0) {
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(mb) : "memory");
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(mb + 8)
                   : "memory");
      asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(mb + 16)
                   : "memory");
    }
    return true;
  }
  { // the partial (a fp32, then m / l with release), then this split's share of
    // the group's combine
    float *const pa = part_o + ((size_t)g * NS + s) * NQ * KV;
#pragma unroll
    for (int qb = 0; qb < QB; qb++) {
      float const ra = rcp(ml[(qb * 8 + t0) * 4 + 1]),
                  rb2 = rcp(ml[(qb * 8 + t0 + 1) * 4 + 1]);
#pragma unroll
      for (int mb_ = 0; mb_ < 4; mb_++) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
          pa[(size_t)(qb * 8 + t0 + (e & 1)) * KV + warp * 64 + mb_ * 16 + g8 +
             (e >> 1) * 8] = acc[mb_][qb][e] * ((e & 1) ? rb2 : ra);
        }
      }
    }
    __syncthreads();
    mla_part_combine<HEADS, HG, KV, NS>(
        rows0, Sp, ml, part_o, part_ml, attn, g, s, nne, mb + 16, sph);
  }
  __syncthreads();
  if (tid == 0) {
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(mb) : "memory");
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(mb + 8)
                 : "memory");
    asm volatile("mbarrier.inval.shared::cta.b64 [%0];" ::"r"(mb + 16)
                 : "memory");
  }
  return true;
}

// task (g, s): Q = mla_absorb (its output [T][heads][KV + ROPE]); grid (heads /
// HG, NS); params = [heads, HG, KV, ROPE, page size, scale as fp32 bits, thr
// (rows: shorter contexts not here), late (optional: the last splits that start
// late, used only when every split gets 3 chunks)]; buffer slots = kv_new
// [T][KV + ROPE] bf16 (mla_norm's), the KV cache [rows][KV + ROPE] bf16, the
// page table [blocks] int32, seq_len_k [1] int32, the partials' outputs
// [groups][NS][HG T][KV] fp32, their m / l [groups][NS][HG T][2] fp32 (0xFF
// each launch); map slot = the KV cache's (host.cuh map_bf16: blocks of 64,
// boxes of 64 rows); the node's output = attn [T][heads][KV] bf16 (0xFF each
// launch)
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class Q,
          class... IN>
__device__ __forceinline__ void run_mla_attn_part(Maps const &maps,
                                                  G const &g,
                                                  KernelLocals &L,
                                                  StaticTask const &tk) {
  constexpr int HEADS = PARAMS::v[0], HG = PARAMS::v[1], KV = PARAMS::v[2],
                ROPE = PARAMS::v[3], PS = PARAMS::v[4];
  constexpr int NS = SELF::y, THR = PARAMS::v[6], LATE = param_or<PARAMS, 7>();
  static_assert(Q::buf >= 0 && SELF::buf >= 0 && SELF::x * HG == HEADS,
                "mla_attn_part: mla_absorb's query, heads / HG groups");
  MlaKv kv;
  kv.kv_new = buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>());
  kv.cache = buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 1>());
  kv.pt = buf_at<int const>(g, slot_at<BUF_SLOTS, 2>());
  kv.seq_k = buf_at<int const>(g, slot_at<BUF_SLOTS, 3>());
#if defined(STATIC_TIMING_BUILD)
  unsigned long long *const stamps = L.stage_stamps;
#else
  unsigned long long *const stamps = nullptr;
#endif
  if (mla_attn_part_task<HEADS, HG, KV, ROPE, PS, NS, LATE>(
          L.sm,
          L.tb,
          stamps,
          &maps.m[slot_at<MAP_SLOTS, 0>()],
          buf_at<__nv_bfloat16 const>(g, Q::buf),
          kv,
          __int_as_float(PARAMS::v[5]),
          buf_at<float>(g, slot_at<BUF_SLOTS, 4>()),
          buf_at<float>(g, slot_at<BUF_SLOTS, 5>()),
          buf_at<__nv_bfloat16>(g, SELF::buf),
          THR,
          tk.x,
          tk.y)) {
    L.ring_relaid = true;
  }
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_mla_attn_part() {
  constexpr int D = PARAMS::v[2] + PARAMS::v[3], NQ = PARAMS::v[1] * T,
                CH = MLA_CH, NBOX = (D / 64 + 1) / 2;
  return 2 * NBOX * 2 * CH * 128 + NQ * D * 2 + T * (2 * D + 16) +
         2 * NQ * CH * 4 + NQ * 4 * 4 + MLA_MAXC * 4 + 24;
}

} // namespace static_mk
