// tasks/attn_residual.cuh -- K3's attention residual after a TP all-reduce
// (KDA layer, the all-reduce's sum fused in): task
// (t, part) of grid (T, 2, 1) = token t, half `part` of its H-wide row; the 2
// halves of a token on the 2 CTAs of a cluster (the compiler's cluster
// pairing), their sums exchanged in shared memory (DSMEM).
//   a   = the GPUs' partial sums of the row (an allreduce_send node's output,
//         bf16, added in fp32 in GPU order)
//   u   = bf16(prefix + a)
//   c   = q * gain (fp32 of bf16 values)
//   s_k = (c . b_k) / sqrt(mean(b_k^2) + eps) for the two bank rows b_0, b_1
//   and
//         u; w = softmax(s_0, s_1, s_u)
//   h   = bf16(w_0 b_0 + w_1 b_1 + w_u u); var = mean of (w . rows)^2 from the
//         row statistics (no second pass)
//   out = bf16(h * rsqrt(var + eps) * ow)
// The bank rows' strides at run time from bank_info [valid rows, token stride,
// row stride] (int32, elements), their number ATTN_RES_NVB at compile time; 2
// rows: the path above with those strides; 1 row: the Gram path (one exchange);
// any other number (3 .. ATTN_RES_NBMAX): the general path: every row's
// statistics before the landing, softmax over all rows and u, the mix in a
// second pass (h = bf16 of the weighted rows, var = mean(h^2) as SGLang's
// reference), the parts' sums exchanged twice. Shared memory: b_0, b_1, u as
// bf16 after the GEMM ring (RING_BYTES), so a GEMM before or after it on the SM
// keeps its ring; q, gain and ow are read from global memory (L2) when needed.
// 256 threads: thread v handles 16-B vectors v, v + 256, ... of the row.
#pragma once
#include "../core.cuh"

// ATTN_RES_NVB=n (1..16, one library per n): a build for layers with n bank
// rows only (bank_info's number must be n): only that path's code is in the
// kernel. The general path's code in the same kernel as the 2-row path made
// every task of the layer ~5 us slower (harness 34.0 vs 29.4, 10-09: the
// kernel's code 27 % larger).
#ifndef ATTN_RES_NVB
#error                                                                         \
    "attn_residual: build with -DATTN_RES_NVB=n (the layer's bank rows, 1..16)"
#endif

namespace static_mk {

constexpr int ATTN_RES_STATS = 9; // ssq0 ssq1 dot0 dot1 g01 | ssqU dotU g0u g1u
constexpr int ATTN_RES_NBMAX = 16; // at most this many bank rows
constexpr int ATTN_RES_GEN =
    2 * ATTN_RES_NBMAX + 2; // general path: ssq_i dot_i of every row, ssqU dotU
static_assert(ATTN_RES_NVB >= 1 && ATTN_RES_NVB <= ATTN_RES_NBMAX,
              "attn_residual: ATTN_RES_NVB 1..16");
// the Gram path (2 < bank rows <= ATTN_RES_GRAM_MAX, or 1): one exchange of
// every row's ssq and dot, the rows' pairwise products, u's ssq and dot and u's
// products with the rows (the output's variance from them: no second exchange,
// no second row read)
constexpr int ATTN_RES_GRAM_MAX = 8;
constexpr int ATTN_RES_GRAM_N =
    2 * ATTN_RES_GRAM_MAX + ATTN_RES_GRAM_MAX * (ATTN_RES_GRAM_MAX - 1) / 2 +
    2 + ATTN_RES_GRAM_MAX;
// the other part writes its sums into this CTA's shared memory, 2 regions
constexpr int ATTN_RES_XPEER = 64;
static_assert(ATTN_RES_GEN <= ATTN_RES_XPEER &&
                  ATTN_RES_GRAM_N <= ATTN_RES_XPEER,
              "attn_residual: an exchange region's values");

template <int H>
__host__ __device__ constexpr int attn_residual_smem() {
  // b0, b1, u (bf16) + per warp the 9 statistics + the 4 weights + (general
  // path) per warp ATTN_RES_GEN sums, the row weights
  return RING_BYTES + 3 * H * 2 + (256 / 32) * ATTN_RES_STATS * 4 + 16 +
         (256 / 32) * ATTN_RES_GEN * 4 + (ATTN_RES_NBMAX + 2) * 4 +
         ATTN_RES_GEN * 4 + 2 * ATTN_RES_XPEER * 4 + 16;
}

template <size_t AOFF, // the sent partial sums' offset in the exchange region
          int H>
__device__ __forceinline__ void attn_residual_task(G const &g,
                                                   KernelLocals &L,
                                                   __nv_bfloat16 const *bank,
                                                   __nv_bfloat16 const *prefix,
                                                   __nv_bfloat16 const *q,
                                                   __nv_bfloat16 const *gain,
                                                   __nv_bfloat16 const *ow,
                                                   __nv_bfloat16 *out,
                                                   float eps,
                                                   int t,
                                                   int part,
                                                   __nv_bfloat16 *u_out,
                                                   int const *bank_info) {
  // part `part` of 2: the row's columns [part HP, part HP + HP); the parts
  // exchange their statistics with the cluster's other CTA and every part
  // reduces them in part order: the same weights on every part
  constexpr int P = 2, HP = H / P;
  constexpr size_t A_RANK = (size_t)T * H * 2; // one GPU's rows in the buffer
  constexpr int NV = HP / 8;                   // 16-B vectors of this part
  static_assert(H % (8 * P) == 0, "attn_residual: whole 16-B vectors per part");
  static_assert(attn_residual_smem<HP>() <= 227 * 1024 - 1024,
                "attn_residual: the row does not fit after the GEMM ring");
  __nv_bfloat16 *b0s = reinterpret_cast<__nv_bfloat16 *>(L.sm + RING_BYTES);
  __nv_bfloat16 *b1s = b0s + HP, *us = b1s + HP;
  float *red = reinterpret_cast<float *>(us + HP); // [8 warps][9]
  float *wts = red + (256 / 32) * ATTN_RES_STATS;  // w0 w1 w2 rstd
  float *redg = wts + 4; // general path: [8 warps][ATTN_RES_GEN]
  float *wg =
      redg +
      (256 / 32) *
          ATTN_RES_GEN; // general path: the rows' weights, then u's, then rstd
  float *gloc =
      wg + ATTN_RES_NBMAX + 2; // general path: the token's sums (thread 0)
  float *xpeer = gloc + ATTN_RES_GEN; // [2][ATTN_RES_XPEER] the other part's
                                      // sums, written by its CTA
  uint32_t const xmb = su32(
      xpeer +
      2 * ATTN_RES_XPEER); // 2 mbarriers, region r's bytes landed (complete_tx)
  // thread 0: v into the other CTA's xpeer region `reg` by st.async, each
  // completing bytes on that CTA's mbarrier `reg` (no fence, no cluster
  // barrier: the 10-10 barrier.cluster.arrive.release version took 1.1 us, the
  // release waiting for this CTA's global stores); cl_get: this CTA's region
  // `reg` once its n values landed
  auto cl_put = [&](float const *v, int n, int reg) {
    uint32_t rank, remote, rbar;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(remote)
                 : "r"(su32(xpeer + reg * ATTN_RES_XPEER)), "r"(rank ^ 1u));
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(rbar)
                 : "r"(xmb + 8u * reg), "r"(rank ^ 1u));
    for (int k = 0; k < n; k++) {
      asm volatile("st.async.shared::cluster.mbarrier::complete_tx::bytes.b32 "
                   "[%0], %1, [%2];" ::"r"(remote + 4u * k),
                   "r"(__float_as_uint(v[k])),
                   "r"(rbar)
                   : "memory");
    }
  };
  auto cl_get = [&](int n, int reg) {
    mbar_expect(xmb + 8u * reg, 4u * n);
    mbar_wait(xmb + 8u * reg, 0);
  };
  // the bank rows: their number at compile time, their strides from bank_info
  constexpr int nvb = ATTN_RES_NVB;
  size_t const bs_tok = (size_t)bank_info[1], bs_row = (size_t)bank_info[2];
  if (bank_info[0] != nvb) {
    __trap(); // the host checks the number
  }
  constexpr bool fast = nvb == 2;
  int const tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  auto bf = [](uint32_t w, int k) {
    float2 const f =
        __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const *>(&w));
    return k ? f.y : f.x;
  };
  auto el = [&](uint4 const &v, int j) {
    uint32_t const w[4] = {v.x, v.y, v.z, v.w};
    return bf(w[j >> 1], j & 1);
  };
  int const c0 = part * HP; // this part's first column
  bank += c0;
  prefix += c0;
  q += c0;
  gain += c0;
  ow += c0;
  out += c0;
  unsigned char const *row = g.rv + AOFF + rg_set() + ((size_t)t * H + c0) * 2;
  // the operand rows in shared memory, copied at the task's start: [prefix row
  // t | q | gain | ow] from the base, in the GEMM ring's region (a GEMM task
  // before this one on the SM is done with it after the barrier below; one
  // after it reloads its stages). Read from global where used, after the
  // launch's L2 flush they came from HBM, one round trip per landed vector
  // group: the layer 1.65 us slower (35.4 vs 33.8, k55; the rows had landed
  // 1.6 us after o_proj's pushes and the loop ran 4.6 us more, k54)
  __nv_bfloat16 *ops_s = reinterpret_cast<__nv_bfloat16 *>(L.sm);
  static_assert(4 * HP * 2 <= RING_BYTES,
                "attn_residual: operand rows in the ring's region");
  __nv_bfloat16 const *pre_r = ops_s, *q_r = ops_s + HP, *g_r = ops_s + 2 * HP,
                      *o_r = ops_s + 3 * HP;
  // the Gram path: the bank rows [nvb][HP] bf16 after the operand rows (the
  // ring's region), per warp the sums [8][GRAM_N], the token's values [GRAM_N]
  // (thread 0) (10-10, aligned eager spans, 2 runs: 1 row 30.1 -> 29.5 us; 3 /
  // 6 / 8 rows LOST +1 / +4.2 / +6.8: the unrolled pair code slows the kernel;
  // so 1 bank row only)
  constexpr bool gram =
      nvb ==
      1; // the Gram path for 1 bank row (other counts but 2: the general path)
  __nv_bfloat16 *const grows = ops_s + 4 * HP;
  float *const gramw =
      reinterpret_cast<float *>(grows + ATTN_RES_GRAM_MAX * HP);
  float *const gvals = gramw + (256 / 32) * ATTN_RES_GRAM_N;
  static_assert(
      (4 + ATTN_RES_GRAM_MAX) * HP * 2 +
              ((256 / 32) + 1) * ATTN_RES_GRAM_N * 4 <=
          RING_BYTES,
      "attn_residual: the Gram path's rows and sums in the ring's region");
  int const npairs = nvb * (nvb - 1) / 2;
  __syncthreads(); // the SM's previous task (a GEMM's MMAs) left the region
  // this CTA in the task (its xpeer free, its mbarriers set): the other CTA
  // waits for it before its first write
  if (tid == 0) {
    mbar_init(xmb, 1);
    mbar_init(xmb + 8, 1);
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
  bool cl_waited =
      false; // the start arrival's wait done (before the first write)
  float p[ATTN_RES_STATS] = {};
  float pu[ATTN_RES_GRAM_MAX] =
      {}; // the Gram path: u . row_i (this thread's vectors)
  // the bank rows (graph inputs, no wait) and their statistics
  for (int v = tid; v < NV; v += 256) {
    uint4 const zero4 = make_uint4(0u, 0u, 0u, 0u);
    uint4 const a = fast ? *reinterpret_cast<uint4 const *>(
                               bank + (size_t)t * bs_tok + 8 * v)
                         : zero4;
    uint4 const b = fast ? *reinterpret_cast<uint4 const *>(
                               bank + (size_t)t * bs_tok + bs_row + 8 * v)
                         : zero4;
    uint4 const qv = *reinterpret_cast<uint4 const *>(q + 8 * v);
    uint4 const gv = *reinterpret_cast<uint4 const *>(gain + 8 * v);
    *reinterpret_cast<uint4 *>(b0s + 8 * v) = a;
    *reinterpret_cast<uint4 *>(b1s + 8 * v) = b;
    *reinterpret_cast<uint4 *>(ops_s + 8 * v) =
        *reinterpret_cast<uint4 const *>(prefix + (size_t)t * H + 8 * v);
    *reinterpret_cast<uint4 *>(ops_s + HP + 8 * v) = qv;
    *reinterpret_cast<uint4 *>(ops_s + 2 * HP + 8 * v) = gv;
    *reinterpret_cast<uint4 *>(ops_s + 3 * HP + 8 * v) =
        *reinterpret_cast<uint4 const *>(ow + 8 * v);
#pragma unroll
    for (int j = 0; j < 8; j++) {
      float const b0 = el(a, j), b1 = el(b, j), c = el(qv, j) * el(gv, j);
      p[0] += b0 * b0;
      p[1] += b1 * b1;
      p[2] += c * b0;
      p[3] += c * b1;
      p[4] += b0 * b1;
    }
  }

  __syncthreads();      // the operand rows written before any thread reads them
  if constexpr (gram) { // the Gram path: every bank row's ssq and dot with c =
                        // q * gain, and the rows' pairwise products (this
    // part's columns), before the landing, in one pass: all rows' vectors
    // loaded together (one memory round trip, not one per row), the rows copied
    // to shared memory for the landing and the output
    constexpr int GM = ATTN_RES_GRAM_MAX, GP = GM * (GM - 1) / 2;
    float ss[GM] = {}, dt[GM] = {}, gp[GP] = {};
    __nv_bfloat16 const *rt = bank + (size_t)t * bs_tok;
    for (int v = tid; v < NV; v += 256) {
      uint4 a[GM];
#pragma unroll
      for (int i = 0; i < GM; i++) {
        if (i < nvb) {
          a[i] =
              *reinterpret_cast<uint4 const *>(rt + (size_t)i * bs_row + 8 * v);
        }
      }
      uint4 const qv = *reinterpret_cast<uint4 const *>(q_r + 8 * v);
      uint4 const gv = *reinterpret_cast<uint4 const *>(g_r + 8 * v);
#pragma unroll
      for (int i = 0; i < GM; i++) {
        if (i < nvb) {
          *reinterpret_cast<uint4 *>(grows + (size_t)i * HP + 8 * v) = a[i];
        }
      }
#pragma unroll
      for (int j = 0; j < 8; j++) {
        float const c = el(qv, j) * el(gv, j);
        float x[GM];
#pragma unroll
        for (int i = 0; i < GM; i++) {
          x[i] = i < nvb ? el(a[i], j) : 0.f;
          ss[i] += x[i] * x[i];
          dt[i] += c * x[i];
        }
        int k = 0;
#pragma unroll
        for (int i = 0; i < GM; i++) {
#pragma unroll
          for (int i2 = i + 1; i2 < GM; i2++, k++) {
            gp[k] += x[i] * x[i2];
          }
        }
      }
    }
    // per warp: ssq, dot to redg (as the general path), the pairs (only those
    // of real rows) to gramw in the path's order
#pragma unroll
    for (int i = 0; i < GM; i++) {
      if (i < nvb) {
        float a0 = ss[i], a1 = dt[i];
#pragma unroll
        for (int d = 16; d >= 1; d >>= 1) {
          a0 += __shfl_xor_sync(0xffffffffu, a0, d);
          a1 += __shfl_xor_sync(0xffffffffu, a1, d);
        }
        if (lane == 0) {
          redg[warp * ATTN_RES_GEN + 2 * i] = a0;
          redg[warp * ATTN_RES_GEN + 2 * i + 1] = a1;
        }
      }
    }
    {
      int k = 0,
          kk = 0; // k: in gp (all GM rows), kk: among the real rows' pairs
#pragma unroll
      for (int i = 0; i < GM; i++) {
#pragma unroll
        for (int i2 = i + 1; i2 < GM; i2++, k++) {
          if (i2 < nvb) {
            float x = gp[k];
#pragma unroll
            for (int d = 16; d >= 1; d >>= 1) {
              x += __shfl_xor_sync(0xffffffffu, x, d);
            }
            if (lane == 0) {
              gramw[warp * ATTN_RES_GRAM_N + 2 * nvb + kk] = x;
            }
            kk++;
          }
        }
      }
    }
  } else if constexpr (!fast) { // general path: every bank row's ssq and dot
                                // with c = q * gain (this part's columns),
                                // before the landing
    for (int i = 0; i < nvb; i++) {
      __nv_bfloat16 const *ri = bank + (size_t)t * bs_tok + (size_t)i * bs_row;
      float ss = 0.f, dt = 0.f;
      for (int v = tid; v < NV; v += 256) {
        uint4 const a = *reinterpret_cast<uint4 const *>(ri + 8 * v);
        if (i < ATTN_RES_GRAM_MAX) { // the row kept for the mix (the second
                                     // pass reads shared memory)
          *reinterpret_cast<uint4 *>(grows + (size_t)i * HP + 8 * v) = a;
        }

        uint4 const qv = *reinterpret_cast<uint4 const *>(q_r + 8 * v);
        uint4 const gv = *reinterpret_cast<uint4 const *>(g_r + 8 * v);
#pragma unroll
        for (int j = 0; j < 8; j++) {
          float const x = el(a, j);
          ss += x * x;
          dt += el(qv, j) * el(gv, j) * x;
        }
      }
#pragma unroll
      for (int d = 16; d >= 1; d >>= 1) {
        ss += __shfl_xor_sync(0xffffffffu, ss, d);
        dt += __shfl_xor_sync(0xffffffffu, dt, d);
      }
      if (lane == 0) {
        redg[warp * ATTN_RES_GEN + 2 * i] = ss;
        redg[warp * ATTN_RES_GEN + 2 * i + 1] = dt;
      }
    }
  }
  // the row of every GPU, u and its statistics: thread v polls its 16-B vector
  // v (then v + 256) of every GPU at once (GPUS loads in flight), then adds the
  // GPUs in GPU order
  if constexpr (REARM) { // the next launch's set: the rows this task lands,
                         // 0xFF again
    unsigned char const *row_next =
        g.rv + AOFF + (EXCHANGE_SET - rg_set()) + ((size_t)t * H + c0) * 2;
    for (int v = tid; v < NV; v += 256) {
      allreduce_rearm<A_RANK>(g, row_next, v);
    }
  }
  // pass 0: the tail code below alone, dry (no output stores, operands not yet
  // valid), while the rows are still in flight, so its instructions are
  // fetched before the landing instead of after it (cold: the layer 0.75 us
  // slower, 44.0 vs 43.2). Pass 1: the landing, then the tail. One code copy
  // of the tail for both passes: the loop is not unrolled.
#pragma unroll 1
  for (int pass = 0; pass < 2; pass++) {
    bool const dry = pass == 0;
    if (!fast && dry) {
      continue; // the general path: no dry pass (its exchanges must not run
                // twice)
    }
    if (!dry) {
      for (int v = tid; v < NV; v += 256) {
        uint4 w[GPUS];
        bool ok[GPUS] = {};
        bool all;
        do {
          all = true;
#pragma unroll
          for (int r = 0; r < GPUS; r++) {
            if (!ok[r]) {
              w[r] = ld16_poll(row + (size_t)r * A_RANK + (size_t)v * 16);
            }
          }
#pragma unroll
          for (int r = 0; r < GPUS; r++) {
            ok[r] = ok[r] || valid16(w[r]);
            all = all && ok[r];
          }
        } while (!all);
#if defined(STATIC_TIMING_BUILD)
        if (tid == 0 && v == tid &&
            L.stage_stamps) { // stage 0: thread 0's first
                              // vector landed from every GPU
          L.stage_stamps[0] = gtime();
        }
#endif
        {
          float s[8] = {};
#pragma unroll
          for (int r = 0; r < GPUS; r++) {
#pragma unroll
            for (int j = 0; j < 8; j++) {
              s[j] += el(w[r], j);
            }
          }
          uint4 const pv = *reinterpret_cast<uint4 const *>(pre_r + 8 * v);
          uint4 const qv = *reinterpret_cast<uint4 const *>(q_r + 8 * v);
          uint4 const gv = *reinterpret_cast<uint4 const *>(g_r + 8 * v);
          uint4 const a = *reinterpret_cast<uint4 const *>(b0s + 8 * v);
          uint4 const bb = *reinterpret_cast<uint4 const *>(b1s + 8 * v);
          float uu[8];
#pragma unroll
          for (int j = 0; j < 8; j++) {
            uu[j] = __bfloat162float(__float2bfloat16(el(pv, j) + s[j]));
            float const c = el(qv, j) * el(gv, j);
            p[5] += uu[j] * uu[j];
            p[6] += c * uu[j];
            p[7] += el(a, j) * uu[j];
            p[8] += el(bb, j) * uu[j];
          }
          if (gram) { // u's products with the bank rows (in shared memory since
                      // before the landing)
#pragma unroll
            for (int i = 0; i < ATTN_RES_GRAM_MAX; i++) {
              if (i < nvb) {
                uint4 const ri = *reinterpret_cast<uint4 const *>(
                    grows + (size_t)i * HP + 8 * v);
#pragma unroll
                for (int j = 0; j < 8; j++) {
                  pu[i] += el(ri, j) * uu[j];
                }
              }
            }
          }
          uint4 const upk =
              pack_bf16x8(make_float4(uu[0], uu[1], uu[2], uu[3]),
                          make_float4(uu[4], uu[5], uu[6], uu[7]));
          *reinterpret_cast<uint4 *>(us + 8 * v) = upk;
          *reinterpret_cast<uint4 *>(u_out + (size_t)t * H + c0 + 8 * v) =
              upk; // the new prefix u (the next layer adds onto it)
        }
      }
    }
#pragma unroll
    for (int k = 0; k < ATTN_RES_STATS; k++) {
      float x = p[k];
#pragma unroll
      for (int d = 16; d >= 1; d >>= 1) {
        x += __shfl_xor_sync(0xffffffffu, x, d);
      }
      if (lane == 0) {
        red[warp * ATTN_RES_STATS + k] = x;
      }
    }
    if (gram && !dry) {
#pragma unroll
      for (int i = 0; i < ATTN_RES_GRAM_MAX; i++) {
        if (i < nvb) {
          float x = pu[i];
#pragma unroll
          for (int d = 16; d >= 1; d >>= 1) {
            x += __shfl_xor_sync(0xffffffffu, x, d);
          }
          if (lane == 0) {
            gramw[warp * ATTN_RES_GRAM_N + 2 * nvb + npairs + 2 + i] = x;
          }
        }
      }
    }
    __syncthreads();
#if defined(STATIC_TIMING_BUILD)
    if (!dry && tid == 0 &&
        L.stage_stamps) { // stage 1: every GPU's row landed, u and
                          // the statistics done (all threads)
      L.stage_stamps[1] = gtime();
    }
#endif
    if constexpr (fast) { // the 2-row path's sums to the other part's CTA
                          // (every thread at the cluster barriers)
      if (!dry) {
        asm volatile("barrier.cluster.wait.aligned;" ::
                         : "memory"); // the other CTA is in this task (its
                                      // start arrival)
        cl_waited = true;
        if (tid == 0) {
          float r[ATTN_RES_STATS] = {};
          for (int w = 0; w < 256 / 32; w++) {
#pragma unroll
            for (int k = 0; k < ATTN_RES_STATS; k++) {
              r[k] += red[w * ATTN_RES_STATS + k];
            }
          }
          cl_put(r, ATTN_RES_STATS, 0);
        }
      }
    }
    if (tid == 0 &&
        fast) { // the softmax weights and the variance, on one thread
      float r[ATTN_RES_STATS] = {};
      for (int w = 0; w < 256 / 32; w++) {
#pragma unroll
        for (int k = 0; k < ATTN_RES_STATS; k++) {
          r[k] += red[w * ATTN_RES_STATS + k];
        }
      }
      if (!dry) { // the other part's sums, landed in this CTA's shared memory,
                  // added in part order
        float ro[ATTN_RES_STATS];
        cl_get(ATTN_RES_STATS, 0);
#pragma unroll
        for (int k = 0; k < ATTN_RES_STATS; k++) {
          ro[k] = xpeer[k];
        }
#pragma unroll
        for (int k = 0; k < ATTN_RES_STATS; k++) { // part 0's, then part 1's
          float const s0 = part == 0 ? r[k] : ro[k],
                      s1 = part == 0 ? ro[k] : r[k];
          r[k] = (0.f + s0) + s1;
        }
      }
      float const ssq0 = r[0], ssq1 = r[1], dot0 = r[2], dot1 = r[3],
                  g01 = r[4], ssqU = r[5], dotU = r[6], g0u = r[7], g1u = r[8];
      float sc[3] = {dot0 * rsqrtf(ssq0 / (float)H + eps),
                     dot1 * rsqrtf(ssq1 / (float)H + eps),
                     dotU * rsqrtf(ssqU / (float)H + eps)};
      float const mx = fmaxf(sc[0], fmaxf(sc[1], sc[2]));
      float den = 0.f;
      for (int k = 0; k < 3; k++) {
        sc[k] = __expf(sc[k] - mx);
        den += sc[k];
      }
      float const rden = __fdividef(1.0f, den);
      float const w0 = sc[0] * rden, w1 = sc[1] * rden, w2 = sc[2] * rden;
      float const var =
          (w0 * w0 * ssq0 + w1 * w1 * ssq1 + w2 * w2 * ssqU +
           2.f * (w0 * w1 * g01 + w0 * w2 * g0u + w1 * w2 * g1u)) /
          (float)H;
      wts[0] = w0;
      wts[1] = w1;
      wts[2] = w2;
      wts[3] = rsqrtf(var + eps);
    }
    __syncthreads();
    if constexpr (!fast) {  // the general path (no dry pass: pass 1 only)
      if constexpr (gram) { // one exchange: every value the weights and the
                            // output's variance need
        int const n = 2 * nvb + npairs + 2 + nvb;
        if (tid == 0) { // this part's sums (warps added in order)
          for (int k = 0; k < n; k++) {
            float x = 0.f;
            if (k < 2 * nvb) {
              for (int w = 0; w < 256 / 32; w++) {
                x += redg[w * ATTN_RES_GEN + k];
              }
            } else if (k == 2 * nvb + npairs || k == 2 * nvb + npairs + 1) {
              for (int w = 0; w < 256 / 32; w++) {
                x += red[w * ATTN_RES_STATS + 5 + (k - 2 * nvb - npairs)];
              }
            } else {
              for (int w = 0; w < 256 / 32; w++) {
                x += gramw[w * ATTN_RES_GRAM_N + k];
              }
            }
            gvals[k] = x;
          }
        }
        // the values to the other part's CTA, theirs added in part order after
        // the cluster barrier
        asm volatile("barrier.cluster.wait.aligned;" ::
                         : "memory"); // the other CTA is in this task (its
                                      // start arrival)
        cl_waited = true;
        if (tid == 0) {
          cl_put(gvals, n, 0);
          cl_get(n, 0);
          for (int k = 0; k < n; k++) {
            float const s0 = part == 0 ? gvals[k] : xpeer[k],
                        s1 = part == 0 ? xpeer[k] : gvals[k];
            gvals[k] = (0.f + s0) + s1;
          }
        }
        if (tid == 0) {
          float const *vals = gvals;
          float const *ssq = vals, *gp = vals + 2 * nvb,
                      *ub = vals + 2 * nvb + npairs + 2;
          float const ssqU = vals[2 * nvb + npairs],
                      dotU = vals[2 * nvb + npairs + 1];
          float mx = dotU * rsqrtf(ssqU / (float)H + eps);
          wg[nvb] = mx;
          for (int i = 0; i < nvb; i++) {
            wg[i] = ssq[2 * i + 1] * rsqrtf(ssq[2 * i] / (float)H + eps);
            mx = fmaxf(mx, wg[i]);
          }
          float den = 0.f;
          for (int i = 0; i <= nvb; i++) {
            wg[i] = __expf(wg[i] - mx);
            den += wg[i];
          }
          float const rden = __fdividef(1.0f, den);
          for (int i = 0; i <= nvb; i++) {
            wg[i] *= rden;
          }
          float const wu = wg[nvb];
          float var = wu * wu * ssqU;
          int k = 0;
          for (int i = 0; i < nvb; i++) {
            var += wg[i] * wg[i] * ssq[2 * i] + 2.f * wg[i] * wu * ub[i];
            for (int j = i + 1; j < nvb; j++, k++) {
              var += 2.f * wg[i] * wg[j] * gp[k];
            }
          }
          wg[ATTN_RES_NBMAX + 1] = rsqrtf(var / (float)H + eps);
        }
        __syncthreads();
        float const rstd_g = wg[ATTN_RES_NBMAX + 1], wu = wg[nvb];
        for (int v = tid; v < NV; v += 256) {
          float acc[8] = {};
#pragma unroll
          for (int i = 0; i < ATTN_RES_GRAM_MAX; i++) {
            if (i < nvb) {
              uint4 const a = *reinterpret_cast<uint4 const *>(
                  grows + (size_t)i * HP + 8 * v);
              float const wi = wg[i];
#pragma unroll
              for (int j = 0; j < 8; j++) {
                acc[j] += wi * el(a, j);
              }
            }
          }
          uint4 const u = *reinterpret_cast<uint4 const *>(us + 8 * v);
          uint4 const o = *reinterpret_cast<uint4 const *>(o_r + 8 * v);
          float y[8];
#pragma unroll
          for (int j = 0; j < 8; j++) {
            y[j] = (acc[j] + wu * el(u, j)) * rstd_g *
                   el(o, j); // h fp32 into the norm (as SGLang's attn_res)
          }
          *reinterpret_cast<uint4 *>(out + (size_t)t * H + 8 * v) =
              pack_bf16x8(make_float4(y[0], y[1], y[2], y[3]),
                          make_float4(y[4], y[5], y[6], y[7]));
        }
        __syncthreads(); // the shared rows are free for the SM's next task
        continue;
      } else {
        if (tid == 0) { // the rows' and u's scores, the softmax weights
          float *loc =
              gloc; // shared memory, not a local array (no stack frame)
          for (int k = 0; k < 2 * nvb; k++) {
            float x = 0.f;
            for (int w = 0; w < 256 / 32; w++) {
              x += redg[w * ATTN_RES_GEN + k];
            }
            loc[k] = x;
          }
          float su = 0.f, du = 0.f;
          for (int w = 0; w < 256 / 32; w++) {
            su += red[w * ATTN_RES_STATS + 5];
            du += red[w * ATTN_RES_STATS + 6];
          }
          loc[2 * nvb] = su;
          loc[2 * nvb + 1] = du;
        }
        { // the scores' sums to the other part's CTA, theirs added in part
          // order after the cluster barrier
          asm volatile("barrier.cluster.wait.aligned;" ::
                           : "memory"); // the other CTA is in this task (its
                                        // start arrival)
          cl_waited = true;
          int const n = 2 * nvb + 2;
          if (tid == 0) {
            cl_put(gloc, n, 0);
            cl_get(n, 0);
            for (int k = 0; k < n; k++) {
              float const s0 = part == 0 ? gloc[k] : xpeer[k],
                          s1 = part == 0 ? xpeer[k] : gloc[k];
              gloc[k] = (0.f + s0) + s1;
            }
          }
        }
        if (tid == 0) {
          float *loc = gloc;
          float mx = -INFINITY;
          for (int i = 0; i <= nvb; i++) {
            float const sc =
                loc[2 * i + 1] * rsqrtf(loc[2 * i] / (float)H + eps);
            wg[i] = sc;
            mx = fmaxf(mx, sc);
          }
          float den = 0.f;
          for (int i = 0; i <= nvb; i++) {
            wg[i] = __expf(wg[i] - mx);
            den += wg[i];
          }
          float const rden = __fdividef(1.0f, den);
          for (int i = 0; i <= nvb; i++) {
            wg[i] *= rden;
          }
        }
        __syncthreads();
        // h = sum_i w_i b_i + w_u u, fp32 (as SGLang's attn_res: not rounded
        // before the norm), into shared memory (b0s + b1s, HP floats: free in
        // the general path), and its sum of squares
        float *hs = reinterpret_cast<float *>(b0s);
        __nv_bfloat16 const *rows = bank + (size_t)t * bs_tok;
        float hh = 0.f;
        for (int v = tid; v < NV; v += 256) {
          float acc[8] = {};
          for (int i = 0; i < nvb; i++) {
            uint4 const a = (nvb <= ATTN_RES_GRAM_MAX)
                                ? *reinterpret_cast<uint4 const *>(
                                      grows + (size_t)i * HP + 8 * v)
                                : *reinterpret_cast<uint4 const *>(
                                      rows + (size_t)i * bs_row + 8 * v);
            float const wi = wg[i];
#pragma unroll
            for (int j = 0; j < 8; j++) {
              acc[j] += wi * el(a, j);
            }
          }
          uint4 const u = *reinterpret_cast<uint4 const *>(us + 8 * v);
          float const wu = wg[nvb];
          float hv[8];
#pragma unroll
          for (int j = 0; j < 8; j++) {
            hv[j] = acc[j] + wu * el(u, j);
            hh += hv[j] * hv[j];
          }
          reinterpret_cast<float4 *>(hs + 8 * v)[0] =
              make_float4(hv[0], hv[1], hv[2], hv[3]);
          reinterpret_cast<float4 *>(hs + 8 * v)[1] =
              make_float4(hv[4], hv[5], hv[6], hv[7]);
        }
#pragma unroll
        for (int d = 16; d >= 1; d >>= 1) {
          hh += __shfl_xor_sync(0xffffffffu, hh, d);
        }
        if (lane == 0) {
          red[warp * ATTN_RES_STATS] = hh;
        }
        __syncthreads();
        if (tid == 0) {
          float sh = 0.f;
          for (int w = 0; w < 256 / 32; w++) {
            sh += red[w * ATTN_RES_STATS];
          }
          // the other part's sum of h^2 (region 1), added in part order
          cl_put(&sh, 1, 1);
          cl_get(1, 1);
          float const so = xpeer[ATTN_RES_XPEER];
          sh = part == 0 ? (0.f + sh) + so : (0.f + so) + sh;
          wg[ATTN_RES_NBMAX + 1] = sh;
        }
        if (tid == 0) {
          wg[ATTN_RES_NBMAX + 1] =
              rsqrtf(wg[ATTN_RES_NBMAX + 1] / (float)H + eps);
        }
        __syncthreads();
        float const rstd_g = wg[ATTN_RES_NBMAX + 1];
        for (int v = tid; v < NV; v += 256) {
          float4 const h0 = reinterpret_cast<float4 const *>(hs + 8 * v)[0],
                       h1 = reinterpret_cast<float4 const *>(hs + 8 * v)[1];
          float const hv[8] = {h0.x, h0.y, h0.z, h0.w, h1.x, h1.y, h1.z, h1.w};
          uint4 const o = *reinterpret_cast<uint4 const *>(o_r + 8 * v);
          float y[8];
#pragma unroll
          for (int j = 0; j < 8; j++) {
            y[j] = hv[j] * rstd_g * el(o, j);
          }
          *reinterpret_cast<uint4 *>(out + (size_t)t * H + 8 * v) =
              pack_bf16x8(make_float4(y[0], y[1], y[2], y[3]),
                          make_float4(y[4], y[5], y[6], y[7]));
        }
        __syncthreads(); // the shared rows are free for the SM's next task
        continue;
      }
    }
    float const w0 = wts[0], w1 = wts[1], w2 = wts[2], rstd = wts[3];
    for (int v = tid; v < NV; v += 256) {
      uint4 const a = *reinterpret_cast<uint4 const *>(b0s + 8 * v);
      uint4 const b = *reinterpret_cast<uint4 const *>(b1s + 8 * v);
      uint4 const u = *reinterpret_cast<uint4 const *>(us + 8 * v);
      uint4 const o = *reinterpret_cast<uint4 const *>(o_r + 8 * v);
      float y[8];
#pragma unroll
      for (int j = 0; j < 8; j++) {
        float const h =
            w0 * el(a, j) + w1 * el(b, j) +
            w2 * el(u, j); // fp32 into the norm (as SGLang's attn_res)
        y[j] = h * rstd * el(o, j);
      }
      if (!dry) {
        *reinterpret_cast<uint4 *>(out + (size_t)t * H + 8 * v) =
            pack_bf16x8(make_float4(y[0], y[1], y[2], y[3]),
                        make_float4(y[4], y[5], y[6], y[7]));
      }
    }
    __syncthreads(); // the shared rows are free for the SM's next task
  }
  if (!cl_waited) { // every arrival waited for (a path without an exchange: the
                    // start arrival's wait here)
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
  }
}

// attn_residual task (t, part): SEND = the pushing gemm_tile (or
// allreduce_send) node whose rows it adds (H = its row width); params = [H, eps
// as fp32 bits, 1 (the compiler's cluster pairing: the token's 2 parts on the 2
// CTAs of a cluster, CTAs 2k, 2k + 1, the kernel launched with clusters of 2)];
// buffer slots = bank, prefix, q, gain, ow (graph tensors read by pointer), u
// (the new prefix out, [T][H] bf16), bank_info [valid rows, token stride, row
// stride] int32
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class SEND,
          class... IN>
__device__ __forceinline__ void run_attn_residual(Maps const &,
                                                  G const &g,
                                                  KernelLocals &L,
                                                  StaticTask const &tk) {
  constexpr int H = PARAMS::v[0];
  static_assert(SEND::buf >= 0 &&
                    exchange_bytes[SEND::buf] == (size_t)GPUS * T * H * 2,
                "attn_residual: every GPU's rows of H in an exchange buffer "
                "(allreduce_send's, or a pushing gemm_tile's)");
  static_assert(SELF::x == T && SELF::y == 2 && SELF::buf >= 0,
                "attn_residual: 2 tasks per token, its output row");
  static_assert(
      param_at<PARAMS, 2>() == 1,
      "attn_residual: the compiler's cluster pairing (params[2] = 1)");
  static_assert(sizeof(BUF_SLOTS::v) / sizeof(int) == 8,
                "attn_residual: 7 buffer slots");
  attn_residual_task<exchange_offset[SEND::buf], H>(
      g,
      L,
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 2>()),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 3>()),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 4>()),
      buf_at<__nv_bfloat16>(g, SELF::buf),
      __int_as_float(PARAMS::v[1]),
      tk.x,
      tk.y,
      buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 5>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 6>()));
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_attn_residual() {
  return attn_residual_smem<PARAMS::v[0] / SELF::y>();
}

} // namespace static_mk
