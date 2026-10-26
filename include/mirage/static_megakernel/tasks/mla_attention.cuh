// tasks/mla_attention.cuh -- Kimi K3 MLA: one head's attention over the
// request's KV rows, its value projection and the output gate (SGLang:
// trtllm-gen MLA decode on the verify step, torch.bmm with w_vc,
// mla_output_gate). Task h of grid (heads, 1, 1) = head h of this GPU:
//   query  = q[t][h] = [q_abs (KV) | q_pe (ROPE)] (mla_absorb's output
//   [T][heads][KV + ROPE] bf16, polled per 16 B) rows   = the request's n =
//   seq_len_k KV rows (KV + ROPE bf16 each): rows j < n - T from the
//   framework's paged KV cache (page
//            table pt: row j at cache row pt[j / PS] * PS + j % PS), the last T
//            rows = this step's tokens (mla_norm's kv_new [T][KV + ROPE],
//            polled per 16 B; the same values it wrote into the cache)
//   attn[t] = softmax_j(scale (query[t] . row_j)) . row_j[:KV] over j <= n - T
//   + t (token t sees the rows before it and the
//            step's tokens up to itself), rounded to bf16 (SGLang's attention
//            writes bf16)
//   v      = bf16(attn . w_vc[h])  w_vc[h] [KV][VD], given transposed: wvcT
//   [heads][VD][KV] bf16 gate   = the g_proj projection's K parts added in part
//   order (a gemm_tile node's output in slots [K parts][T][heads VD]
//            fp32, 0xFF-prefilled), head h's VD columns, rounded to bf16
//   out[t][h VD + d] = bf16(v * bf16(sigmoid(gate)))  -> the node's output
//   [T][heads VD] bf16 (o_proj's input: 0xFF before
//            each launch, polled by the warp-MMA tile)
// The rows go through shared memory 64 at a time (CH): scores S [T][CH] by
// mma.sync (M = 16 rows, N = 8 tokens, K = KV + ROPE), an online softmax per
// token (running max m, sum l; the output rescaled by exp(m_old - m_new) per
// chunk), then the output O^T [KV][T] += rows^T . P^T (M = 16 of the KV dims
// via ldmatrix.trans, N = 8 tokens, K = the chunk's rows; P in bf16). KV splits
// (grid y = NS > 1): task (h, s) runs split s's share of the chunks; s > 0
// writes its partial (the unnormalized output, then m and l with release), s =
// 0 runs its share, then waits for the others' m / l (acquire), adds their
// rescaled outputs (splits with no rows skipped) and finishes the head. Long
// contexts (n > thr rows): the attention comes from mla_attn_part (head groups
// over KV splits; its output attn [T][heads][KV] bf16, 0xFF-prefilled): split 0
// polls head h's rows and runs the same value projection and gate; splits > 0
// exit at once. (Per-head tasks read the KV rows once per head and run 8
// queries per chunk: at a short context one hand-off fewer and less cold
// code, 38.2 vs 43.6 us at 37 rows; at a long one the groups read them 3 times
// in all, 53.7 vs 70.4 us at 4000 rows, 10-11.) The head's w_vc (VD x KV bf16,
// 128 KB) is copied into shared memory as the task starts (split 0). Whole
// shared memory: a GEMM after it on the SM starts from a fresh ring
// (L.ring_relaid).
#pragma once
#include "../core.cuh"
#include "gemm_tile.cuh" // ldsm_x4, ldsm_x2, mma_16816, valid16h
#include "mla_common.cuh"

namespace static_mk {

__device__ __forceinline__ void
    ldsm_x4_t(uint32_t a, uint32_t (&r)[4]) { // transposed 8x8 matrices
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
      : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
      : "r"(a));
}

struct MlaKv { // the framework's KV cache and this step's view of it
  __nv_bfloat16 const *cache;  // [rows][KV + ROPE] bf16 (the layer's buffer)
  __nv_bfloat16 const *kv_new; // [T][KV + ROPE] bf16 (mla_norm)
  int const *pt;               // the request's page table [blocks] int32
  int const *seq_k; // [1] int32: the KV rows incl. this step's T tokens
};

// a thread's 16-B vectors (bit i of pend: vector i) copied from src(i) to
// dst(i) until every 16-bit value is written (no 0xFFFF): one round trip per
// round (cp.async, every outstanding group waited), checked in shared memory
// (no registers held)
template <int PER, class SRC, class DST>
__device__ __forceinline__ void poll16_smem(uint32_t pend, SRC src, DST dst) {
  while (pend != 0u) {
#pragma unroll
    for (int i = 0; i < PER; i++) {
      if ((pend >> i) & 1u) {
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(su32(dst(i))),
            "l"(src(i))
            : "memory");
      }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#pragma unroll 1
    for (int i = 0; i < PER; i++) {
      if (((pend >> i) & 1u) &&
          valid16h(*reinterpret_cast<uint4 const *>(dst(i)))) {
        pend &= ~(1u << i);
      }
    }
  }
}

struct MlaSplit { // the KV splits' partial results (task (h, s > 0) -> task (h,
                  // 0))
  float *o;       // [heads][NS][T][KV] fp32: the unnormalized output (no reset:
                  // written before its m / l)
  float *ml; // [heads][NS][T][2] fp32: the running max and sum (0xFF before
             // each launch; written last, with release)
};

template <int NSPLIT, int HEADS, int KV, int ROPE, int VD, int PS, int NS>
__device__ __forceinline__ void mla_attention_task(KernelLocals &L,
                                                   __nv_bfloat16 const *q,
                                                   MlaKv const kv,
                                                   __nv_bfloat16 const *wvcT,
                                                   float const *gpart,
                                                   float scale,
                                                   __nv_bfloat16 *out,
                                                   MlaSplit const sp,
                                                   __nv_bfloat16 const *attn_in,
                                                   int thr,
                                                   int h,
                                                   int s) {
  static_assert(KV == 512 && ROPE == 64 && VD == 128,
                "mla_attention: the K3 shapes");
  constexpr int D = KV + ROPE; // 576: a row
  constexpr int RB = D * 2; // 1152 B per row in shared memory (72 chunks of 16
                            // B, swizzled within each 8)
  constexpr int CH = 64;    // rows per chunk
  constexpr int WVR = KV * 2;    // 1 KB per w_vc row
  constexpr int GN = HEADS * VD; // g_proj's row
  static_assert(T == 8, "8 tokens: the MMAs' N, one softmax warp per token");
  int const tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  char *const sm = L.sm;
  // layout: wvcT [VD][KV] (128 KB) | rows [CH][D] (72 KB) | Qs [T][D] (9 KB) |
  // P [T][CH] bf16 (1 KB) | S [T][CH] fp32 (2 KB)
  //         | ml [T][4] fp32: m, l, the chunk's rescale | gate [T][VD] fp32 (4
  //         KB) | Sp [T][CH] fp32 (2 KB) ; O staging [T][KV] bf16 reuses rows
  uint32_t const wv = su32(sm);
  char *const rows = sm + VD * WVR;
  uint32_t const rbase = su32(rows);
  char *const qs = rows + CH * RB;
  uint32_t const qbase = su32(qs);
  __nv_bfloat16 *const P = reinterpret_cast<__nv_bfloat16 *>(qs + T * RB);
  float *const S = reinterpret_cast<float *>(P + T * CH);
  float *const ml = S + T * CH;
  float *const gs = ml + T * 4;
  float *const Sp = gs + T * VD; // [T][CH] fp32: the upper k-half's scores
  int const n = *kv.seq_k, L0 = n - T; // rows before this step's tokens
  // this split's chunks [cs, ce) of the nch 64-row chunks, rounded up: split 0
  // always has the first (a short context runs in split 0 alone, no hand-off)
  int const nch = (n + CH - 1) / CH, cs = (s * nch + NS - 1) / NS,
            ce = ((s + 1) * nch + NS - 1) / NS;
  bool const grouped = n > thr; // the attention from mla_attn_part
  if (grouped && s > 0) {
    return;
  }
  if (s > 0 && cs >= ce) { // a split with no rows: its empty partial (m -inf, l
                           // 0) at once, without waiting for the query (the
    // task after it on the SM, e.g. o_proj, starts its weight copy now: 10-11,
    // an empty split waiting for the query delayed o_proj's input by 4.5 us)
    if (tid < T) {
      float *const pml = sp.ml + (((size_t)h * NS + s) * T + tid) * 2;
      asm volatile("st.release.gpu.global.v2.f32 [%0], {%1, %2};" ::"l"(pml),
                   "f"(-INFINITY),
                   "f"(0.f)
                   : "memory");
    }
    return;
  }
  __syncthreads(); // the SM's previous task is done with the shared memory
  if (s == 0) {    // the head's w_vc^T: VD rows of KV, by cp.async
    __nv_bfloat16 const *src = wvcT + (size_t)h * VD * KV;
    for (int i = tid; i < VD * (WVR / 16); i += 256) {
      int const r = i / (WVR / 16), c = i % (WVR / 16);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(
                       wv + r * WVR + ((c ^ (r & 7)) << 4)),
                   "l"(src + (size_t)r * KV + c * 8)
                   : "memory");
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  }
  if (tid < T) {
    ml[tid * 4 + 0] = -INFINITY;
    ml[tid * 4 + 1] = 0.f;
  }
  static_assert(CH <= PS && PS % CH == 0,
                "mla_attention: a chunk within one page");
  // chunk ci's rows that need no input of this kernel: the cache rows before
  // this step (cp.async, one group) and the rows >= n (zero). Chunk 0's are
  // issued here, as the task starts (the cache rows come from HBM: 3 us after
  // the query landed, 10-11)
  auto issue_cache_rows = [&](int ci) {
    int const j0 = ci * CH;
    int const page =
        j0 < L0 ? kv.pt[j0 / PS] : 0; // the chunk's page, read once
    for (int i = tid; i < CH * (D / 8); i += 256) {
      int const r = i / (D / 8), c = i % (D / 8), j = j0 + r;
      uint32_t const dst = rbase + r * RB + (((c & ~7) | ((c ^ r) & 7)) << 4);
      if (j < L0) {
        __nv_bfloat16 const *src =
            kv.cache + ((size_t)page * PS + j % PS) * D + c * 8;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst),
                     "l"(src)
                     : "memory");
      } else if (j >= n) {
        *reinterpret_cast<uint4 *>(rows + (dst - rbase)) =
            make_uint4(0u, 0u, 0u, 0u);
      }
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
  };
  if (!grouped && cs < ce) {
    issue_cache_rows(cs);
  }
  // the gate of head h (polled first: ready long before the query, its round
  // trip hidden behind the query's wait; at the attention's end it cost ~1 us
  // on the critical path, 10-12 A/B) (its parts added, bf16, then
  // bf16(sigmoid)): T x VD
  for (int v = tid; s == 0 && v < T * VD / 4; v += 256) { // (split 0 only)
    int const t = v / (VD / 4), c = (v % (VD / 4)) * 4;
    float const *p0 = gpart + (size_t)t * GN + h * VD + c;
    float4 zp[NSPLIT];
    bool all;
    do {
#pragma unroll
      for (int p = 0; p < NSPLIT; p++) {
        zp[p] = ldf4_relaxed(p0 + (size_t)p * T * GN);
      }
      all = true;
#pragma unroll
      for (int p = 0; p < NSPLIT; p++) {
        all = all && validf4(zp[p]);
      }
    } while (!all);
    float z[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int p = 0; p < NSPLIT; p++) {
      z[0] += zp[p].x;
      z[1] += zp[p].y;
      z[2] += zp[p].z;
      z[3] += zp[p].w;
    }
    if constexpr (REARM) {
#pragma unroll
      for (int p = 0; p < NSPLIT; p++) {
        rearm16(p0 + (size_t)p * T * GN);
      }
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
      float const gb = __bfloat162float(__float2bfloat16(z[j]));
      gs[t * VD + c + j] =
          __bfloat162float(__float2bfloat16(1.f / (1.f + __expf(-gb))));
    }
  }
  __nv_bfloat16 *const os = reinterpret_cast<__nv_bfloat16 *>(
      rows);     // [T][KV] bf16, 1-KB rows, swizzled (as w_vc's)
  if (grouped) { // head h's attention rows from mla_attn_part -> os (the w_vc
                 // copy waited with them)
    constexpr int NV = T * (KV / 8), PER = (NV + 255) / 256;
    uint32_t pend = 0;
#pragma unroll
    for (int i = 0; i < PER; i++) {
      pend |= (tid + 256 * i < NV ? 1u : 0u) << i;
    }
    poll16_smem<PER>(
        pend,
        [&](int i) -> void const * {
          int const v = tid + 256 * i, t = v / (KV / 8), c = v % (KV / 8);
          return attn_in + ((size_t)t * HEADS + h) * KV + c * 8;
        },
        [&](int i) -> char * {
          int const v = tid + 256 * i, t = v / (KV / 8), c = v % (KV / 8);
          return reinterpret_cast<char *>(os + t * KV) +
                 (((c & ~7) | ((c ^ t) & 7)) << 4);
        });
#if defined(STATIC_TIMING_BUILD)
    __syncthreads();
    if (tid == 0 && L.stage_stamps) {
      L.stage_stamps[0] = gtime();
      L.stage_stamps[1] = gtime();
    }
#endif
  } else {
    // the query of head h -> Qs (row t, 16-B chunk c at (c & ~7) | ((c ^ t) &
    // 7)); a thread's vectors polled together
    {
      constexpr int NQV = T * (D / 8), PER = (NQV + 255) / 256;
      uint4 x[PER];
      bool got[PER] = {};
      bool all;
      do {
        all = true;
#pragma unroll
        for (int i = 0; i < PER; i++) {
          int const v = tid + 256 * i;
          if (v < NQV && !got[i]) {
            x[i] = ld16_poll(reinterpret_cast<unsigned char const *>(
                q + ((size_t)(v / (D / 8)) * HEADS + h) * D +
                (v % (D / 8)) * 8));
          }
        }
#pragma unroll
        for (int i = 0; i < PER; i++) {
          int const v = tid + 256 * i;
          if (v < NQV && !got[i]) {
            got[i] = valid16h(x[i]);
            all = all && got[i];
          }
        }
      } while (!all);
#pragma unroll
      for (int i = 0; i < PER; i++) {
        int const v = tid + 256 * i;
        if (v < NQV) {
          int const t = v / (D / 8), c = v % (D / 8);
          *reinterpret_cast<uint4 *>(qs + t * RB +
                                     (((c & ~7) | ((c ^ t) & 7)) << 4)) = x[i];
        }
      }
    }
#if defined(STATIC_TIMING_BUILD)
    __syncthreads();
    if (tid == 0 && L.stage_stamps) { // stage 0: the query and the gate landed
      L.stage_stamps[0] = gtime();
    }
#endif
    // the output accumulator O^T [KV][T]: warp w KV dims [64 w, 64 w + 64) (4
    // blocks of 16), per thread 4 x 4
    float acc[4][4] = {};
    for (int ci = cs; ci < ce; ci++) {
      int const j0 = ci * CH;
      __syncthreads(); // the previous chunk's rows are no longer read
      // the chunk's rows: cache rows by cp.async, rows >= n zero; then this
      // step's rows (at most T in the chunk) polled from kv_new, a thread's (at
      // most 3) vectors together
      if (ci > cs) {
        issue_cache_rows(ci);
      }
      {
        int const rs = max(L0 - j0, 0),
                  re = min(n - j0,
                           CH); // this step's rows in the chunk: [rs, re)
        int const nv = re > rs ? (re - rs) * (D / 8) : 0;
        constexpr int PER = (T * (D / 8) + 255) / 256;
        uint4 x[PER];
        bool got[PER];
#pragma unroll
        for (int k = 0; k < PER; k++) {
          got[k] = tid + 256 * k >= nv;
        }
        bool all;
        do {
          all = true;
#pragma unroll
          for (int k = 0; k < PER; k++) {
            if (!got[k]) {
              int const v = tid + 256 * k, r = rs + v / (D / 8),
                        c = v % (D / 8);
              x[k] = ld16_poll(reinterpret_cast<unsigned char const *>(
                  kv.kv_new + (size_t)(j0 + r - L0) * D + c * 8));
            }
          }
#pragma unroll
          for (int k = 0; k < PER; k++) {
            if (!got[k]) {
              if (valid16h(x[k])) {
                int const v = tid + 256 * k, r = rs + v / (D / 8),
                          c = v % (D / 8);
                *reinterpret_cast<uint4 *>(
                    rows + r * RB + ((((c & ~7) | ((c ^ r) & 7))) << 4)) = x[k];
                got[k] = true;
              } else {
                all = false;
              }
            }
          }
        } while (!all);
      }
      asm volatile("cp.async.wait_group 0;" ::
                       : "memory"); // the chunk's cache rows (and w_vc's copy,
                                    // issued first)
      __syncthreads();
      // scores: warp w rows [16 (w & 3), + 16) of the chunk, the half (w >> 2)
      // of the D / 16 k-steps (2 accumulators: even / odd steps); the upper
      // half's sums go through Sp, the lower half adds them, scales and masks
      // into S
      {
        constexpr int KS = D / 16, KH = KS / 2;
        float s0[4] = {0.f, 0.f, 0.f, 0.f}, s1[4] = {0.f, 0.f, 0.f, 0.f};
        int const rb = warp & 3, k0 = (warp >> 2) * KH;
#pragma unroll 3
        for (int q2 = 0; q2 < KH; q2++) {
          int const ks = k0 + q2;
          uint32_t b0, b1;
          {
            int const t = lane & 7, c = 2 * ks + ((lane >> 3) & 1);
            ldsm_x2(qbase + t * RB + (((c & ~7) | ((c ^ t) & 7)) << 4), b0, b1);
          }
          int const r = rb * 16 + (lane & 7) + ((lane >> 3) & 1) * 8,
                    c = 2 * ks + (lane >> 4);
          uint32_t a[4];
          ldsm_x4(rbase + r * RB + (((c & ~7) | ((c ^ r) & 7)) << 4), a);
          if (q2 & 1) {
            mma_16816(s1, a, b0, b1);
          } else {
            mma_16816(s0, a, b0, b1);
          }
        }
        int const g = lane >> 2, t0 = 2 * (lane & 3), r0 = rb * 16 + g;
        if (warp >= 4) {
#pragma unroll
          for (int e = 0; e < 4; e++) {
            Sp[(t0 + (e & 1)) * CH + r0 + (e >> 1) * 8] = s0[e] + s1[e];
          }
        }
        __syncthreads();
        if (warp < 4) {
#pragma unroll
          for (int e = 0; e < 4;
               e++) { // e: row r0 (+ 8 for e >= 2), token t0 (+ 1 for odd e)
            int const r = r0 + (e >> 1) * 8, t = t0 + (e & 1), j = j0 + r;
            float const v = (s0[e] + s1[e]) + Sp[t * CH + r];
            S[t * CH + r] = j <= L0 + t
                                ? v * scale
                                : -INFINITY; // (also j >= n: L0 + t < n)
          }
        }
      }
      __syncthreads();
      // online softmax: warp t = token t (2 values per lane)
      {
        int const t = warp;
        float const x0 = S[t * CH + lane], x1 = S[t * CH + lane + 32];
        float mx = fmaxf(x0, x1);
#pragma unroll
        for (int d = 16; d >= 1; d >>= 1) {
          mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, d));
        }
        float const m_old = ml[t * 4], m_new = fmaxf(m_old, mx);
        float const p0 = x0 == -INFINITY ? 0.f : __expf(x0 - m_new),
                    p1 = x1 == -INFINITY ? 0.f : __expf(x1 - m_new);
        __nv_bfloat16 const b0 = __float2bfloat16(p0),
                            b1 = __float2bfloat16(p1);
        // P [T][CH] bf16, rows of 128 B (8 chunks, swizzled by the token)
        int const c0 = lane >> 3, c1 = (lane + 32) >> 3;
        P[t * CH + (((c0 ^ t) & 7) << 3) + (lane & 7)] = b0;
        P[t * CH + (((c1 ^ t) & 7) << 3) + (lane & 7)] = b1;
        float sum = p0 + p1; // l over the unrounded exps (SGLang's attention:
                             // bf16 P for P.V, fp32 l)
#pragma unroll
        for (int d = 16; d >= 1; d >>= 1) {
          sum += __shfl_xor_sync(0xffffffffu, sum, d);
        }
        __syncwarp();
        if (lane == 0) {
          float const resc = m_old == -INFINITY ? 0.f : __expf(m_old - m_new);
          ml[t * 4] = m_new;
          ml[t * 4 + 1] = ml[t * 4 + 1] * resc + sum;
          ml[t * 4 + 2] = resc;
        }
      }
      __syncthreads();
      { // O^T [KV][T] = O^T * resc + rows^T [KV][CH] . P^T
        int const t0 = 2 * (lane & 3);
        float const ra = ml[t0 * 4 + 2], rb = ml[(t0 + 1) * 4 + 2];
#pragma unroll
        for (int mb = 0; mb < 4; mb++) {
          acc[mb][0] *= ra;
          acc[mb][1] *= rb;
          acc[mb][2] *= ra;
          acc[mb][3] *= rb;
        }
        uint32_t const pbase = su32(P);
#pragma unroll
        for (int ks = 0; ks < CH / 16; ks++) {
          uint32_t b0, b1;
          {
            int const t = lane & 7, c = 2 * ks + ((lane >> 3) & 1);
            ldsm_x2(pbase + t * (CH * 2) + (((c ^ t) & 7) << 4), b0, b1);
          }
#pragma unroll
          for (int mb = 0; mb < 4; mb++) {
            // A = rows^T: 8x8 matrix i from rows (k) ks * 16 + (i >> 1) * 8 +
            // (lane & 7), dims (m) of chunk (i & 1)
            int const i = lane >> 3, r = ks * 16 + (i >> 1) * 8 + (lane & 7);
            int const c = (warp * 64 + mb * 16) / 8 +
                          (i & 1); // 16-B chunk of the KV dims
            uint32_t a[4];
            ldsm_x4_t(rbase + r * RB + (((c & ~7) | ((c ^ r) & 7)) << 4), a);
            mma_16816(acc[mb], a, b0, b1);
          }
        }
      }
    }
    __syncthreads(); // the last chunk's rows are no longer read: the region
                     // stages O
    if (s > 0) { // a partial: the unnormalized output (only if this split had
                 // rows), then m / l (release: after the output)
      float *const po = sp.o + ((size_t)h * NS + s) * T * KV;
      if (cs < ce) {
        int const g = lane >> 2, t0 = 2 * (lane & 3);
#pragma unroll
        for (int mb = 0; mb < 4; mb++) {
#pragma unroll
          for (int e = 0; e < 4; e++) {
            po[(t0 + (e & 1)) * KV + warp * 64 + mb * 16 + g + (e >> 1) * 8] =
                acc[mb][e];
          }
        }
      }
      __syncthreads();
      if (tid < T) {
        float *const pml = sp.ml + (((size_t)h * NS + s) * T + tid) * 2;
        float const m = ml[tid * 4], l = ml[tid * 4 + 1];
        asm volatile("st.release.gpu.global.v2.f32 [%0], {%1, %2};" ::"l"(pml),
                     "f"(m),
                     "f"(l)
                     : "memory");
      }
      __syncthreads();
      L.ring_relaid = true;
      return;
    }
    if (NS > 1 &&
        ce < nch) { // split 0, other splits with rows: their partials, combined
                    // (max over the splits' m, their sums
      // and outputs rescaled to it; splits with no rows: l = 0, skipped). Not
      // entered when split 0 has every chunk: the block's code is then never
      // fetched (10-11: run once per launch, cold in the instruction cache, it
      // cost 3.3 us at 37 rows)
      float(*const cmb)[NS][2] = reinterpret_cast<float(*)[NS][2]>(
          S); // [T][NS][2] per token and split: m, l (S is free now)
      static_assert(T * NS * 2 <= T * CH,
                    "mla_attention: the splits' m / l in S's region");
      if (tid < T * NS) {
        int const t = tid / NS, s2 = tid % NS;
        float m = ml[t * 4], l = ml[t * 4 + 1];
        if (s2 > 0 &&
            (s2 * nch + NS - 1) / NS >=
                ((s2 + 1) * nch + NS - 1) /
                    NS) { // an empty split (no rows): nothing to wait for
          m = -INFINITY, l = 0.f;
        } else if (s2 > 0) {
          float const *pml = sp.ml + (((size_t)h * NS + s2) * T + t) * 2;
          float2 v;
          do {
            asm volatile("ld.acquire.gpu.global.v2.f32 {%0, %1}, [%2];"
                         : "=f"(v.x), "=f"(v.y)
                         : "l"(pml)
                         : "memory");
          } while (!validf(v.x) || !validf(v.y));
          m = v.x, l = v.y;
        }
        cmb[t][s2][0] = m;
        cmb[t][s2][1] = l;
      }
      __syncthreads();
      int const g = lane >> 2, t0 = 2 * (lane & 3);
#pragma unroll
      for (int e2 = 0; e2 < 2;
           e2++) { // tokens t0, t0 + 1: the global max, split 0's factor; the
                   // total sum into ml[t][1]
        int const t = t0 + e2;
        float M = -INFINITY;
#pragma unroll
        for (int s2 = 0; s2 < NS; s2++) {
          M = fmaxf(M, cmb[t][s2][0]);
        }
        float const f0 =
            cmb[t][0][0] == -INFINITY ? 0.f : __expf(cmb[t][0][0] - M);
#pragma unroll
        for (int mb = 0; mb < 4; mb++) {
          acc[mb][e2] *= f0;
          acc[mb][e2 + 2] *= f0;
        }
        float Lt = cmb[t][0][1] * f0;
        for (int s2 = 1; s2 < NS; s2++) {
          float const l2 = cmb[t][s2][1];
          if (l2 == 0.f) {
            continue;
          }
          float const f = __expf(cmb[t][s2][0] - M);
          Lt += l2 * f;
          float const *po = sp.o + (((size_t)h * NS + s2) * T + t) * KV;
#pragma unroll
          for (int mb = 0; mb < 4; mb++) {
            int const d = warp * 64 + mb * 16 + g;
            acc[mb][e2] += __ldcg(po + d) * f;
            acc[mb][e2 + 2] += __ldcg(po + d + 8) * f;
          }
        }
        __syncthreads(); // (every thread has read cmb[t][.][1] for its tokens
                         // before thread 0 of the warp rewrites ml)
        if (lane < 4 && warp == 0) {
          ml[t * 4 + 1] = Lt;
        }
        __syncthreads();
      }
    }
#if defined(STATIC_TIMING_BUILD)
    if (tid == 0 && L.stage_stamps) {
      L.stage_stamps[1] = gtime();
    }
#endif

    {
      int const g = lane >> 2, t0 = 2 * (lane & 3);
      float const la = 1.f / ml[t0 * 4 + 1], lb = 1.f / ml[(t0 + 1) * 4 + 1];
#pragma unroll
      for (int mb = 0; mb < 4; mb++) {
#pragma unroll
        for (int e = 0; e < 4; e++) {
          int const d = warp * 64 + mb * 16 + g + (e >> 1) * 8,
                    t = t0 + (e & 1);
          int const c = d >> 3;
          os[t * KV + (((c & ~7) | ((c ^ t) & 7)) << 3) + (d & 7)] =
              __float2bfloat16(acc[mb][e] * ((e & 1) ? lb : la));
        }
      }
    }
  }
  __syncthreads();
  // v^T [VD][T] = wvcT [VD][KV] . attn^T: warp w rows [16 w, 16 w + 16), KV /
  // 16 k-steps
  float va[4] = {0.f, 0.f, 0.f, 0.f},
        vb[4] = {
            0.f, 0.f, 0.f, 0.f}; // even / odd k-steps (2 dependency chains)
  uint32_t const obase = su32(os);
#pragma unroll 4
  for (int ks = 0; ks < KV / 16; ks++) {
    uint32_t b0, b1;
    {
      int const t = lane & 7, c = 2 * ks + ((lane >> 3) & 1);
      ldsm_x2(obase + t * (KV * 2) + (((c & ~7) | ((c ^ t) & 7)) << 4), b0, b1);
    }
    int const r = warp * 16 + (lane & 7) + ((lane >> 3) & 1) * 8,
              c = 2 * ks + (lane >> 4);
    uint32_t a[4];
    ldsm_x4(wv + r * WVR + (((c & ~7) | ((c ^ r) & 7)) << 4), a);
    if (ks & 1) {
      mma_16816(vb, a, b0, b1);
    } else {
      mma_16816(va, a, b0, b1);
    }
  }
#pragma unroll
  for (int e = 0; e < 4; e++) {
    va[e] += vb[e];
  }
  // out = bf16(bf16(v) * gate) -> stage [T][VD] bf16 in S's region, then 16-B
  // stores
  __nv_bfloat16 *const ost =
      reinterpret_cast<__nv_bfloat16 *>(qs); // Qs no longer read
  {
    int const g = lane >> 2, t0 = 2 * (lane & 3);
#pragma unroll
    for (int e = 0; e < 4; e++) {
      int const d = warp * 16 + g + (e >> 1) * 8, t = t0 + (e & 1);
      float const vb = __bfloat162float(__float2bfloat16(va[e]));
      ost[t * VD + d] = __float2bfloat16(vb * gs[t * VD + d]);
    }
  }
  __syncthreads();
  if (tid < T * VD / 8) {
    int const t = tid / (VD / 8), c = (tid % (VD / 8)) * 8;
    *reinterpret_cast<uint4 *>(out + (size_t)t * HEADS * VD + h * VD + c) =
        *reinterpret_cast<uint4 const *>(ost + t * VD + c);
  }
  __syncthreads();
  L.ring_relaid = true;
}

// task (h, s): Q = mla_absorb (its output [T][heads][KV + ROPE]), GP = the
// gemm_tile node of g_proj (K parts in slots, N = heads VD), AP = mla_attn_part
// (its output attn [T][heads][KV] bf16: the attention of contexts over thr
// rows); grid y = NS KV splits (split s: chunks [ceil(s nch / NS), ceil((s + 1)
// nch / NS)) of the nch 64-row chunks; split 0 combines the others and writes
// the head's output); params = [heads, KV, ROPE, VD, page size, scale as fp32
// bits, thr]; buffer slots = wvcT [heads][VD][KV] bf16, kv_new [T][KV + ROPE]
// bf16 (mla_norm's), the KV cache [rows][KV + ROPE] bf16, the page table
// [blocks] int32, seq_len_k [1] int32; NS > 1: the partials' outputs
// [heads][NS][T][KV] fp32, their m / l [heads][NS][T][2] fp32 (0xFF before each
// launch)
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class Q,
          class GP,
          class WV,
          class AP,
          class... IN>
__device__ __forceinline__ void run_mla_attention(Maps const &,
                                                  G const &g,
                                                  KernelLocals &L,
                                                  StaticTask const &tk) {
  constexpr int
      NS = SELF::y,
      THR = PARAMS::v[6]; // KV splits (grid y), the grouped path's context rows
  constexpr int HEADS = PARAMS::v[0], KV = PARAMS::v[1], ROPE = PARAMS::v[2],
                VD = PARAMS::v[3], PS = PARAMS::v[4];
  static_assert(Q::buf >= 0 && GP::buf >= 0 && GP::v[0] == COMBINE_SLOTS &&
                    GP::v[2] == HEADS * VD,
                "mla_attention: mla_absorb's query, g_proj's K parts in slots "
                "(N = heads x VD)");
  static_assert(SELF::x == HEADS && SELF::buf >= 0 && NS <= 32,
                "mla_attention: one task per head and KV split, its output");
  static_assert(
      AP::buf >= 0,
      "mla_attention: mla_attn_part's output (the long contexts' attention)");
  MlaSplit sp{};
  if constexpr (NS > 1) {
    sp.o = buf_at<float>(g, slot_at<BUF_SLOTS, 5>());
    sp.ml = buf_at<float>(g, slot_at<BUF_SLOTS, 6>());
  }
  MlaKv kv;
  kv.kv_new = buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 1>());
  kv.cache = buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 2>());
  kv.pt = buf_at<int const>(g, slot_at<BUF_SLOTS, 3>());
  kv.seq_k = buf_at<int const>(g, slot_at<BUF_SLOTS, 4>());
  mla_attention_task<GP::y, HEADS, KV, ROPE, VD, PS, NS>(
      L,
      buf_at<__nv_bfloat16 const>(g, Q::buf),
      kv,
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<float const>(g, GP::buf),
      __int_as_float(PARAMS::v[5]),
      buf_at<__nv_bfloat16>(g, SELF::buf),
      sp,
      buf_at<__nv_bfloat16 const>(g, AP::buf),
      THR,
      tk.x,
      tk.y);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_mla_attention() {
  constexpr int KV = PARAMS::v[1], D = PARAMS::v[1] + PARAMS::v[2],
                VD = PARAMS::v[3];
  return VD * KV * 2 + 64 * D * 2 + T * D * 2 + T * 64 * 2 + T * 64 * 4 +
         T * 4 * 4 + T * VD * 4 + T * 64 * 4;
}

} // namespace static_mk
