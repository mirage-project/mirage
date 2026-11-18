// tasks/indexer_score.cuh -- DeepSeek-V4.1's indexer scores (model.py
// L571-575, Indexer.forward): task (t, c) of grid (T, NCH): token t,
// compressed positions [CH c, CH c + CH) of its n = (p + 1) / R, CH = 256 x
// the halves per task, the fewest that make NCH tasks (the plan's). Per
// position k: score[k] = bf16(sum over the H = 32 heads of bf16(relu(bf16(q_h .
// k)) x w_h)), where q_h is indexer_q's output (fp4 values as bf16), k the
// index K row (e2m1 x its group's power-of-2 scale, exact in bf16) and w_h =
// bf16(weights_proj's K parts added) x 2^-6 (softmax scale 128^-0.5 x 32^-0.5,
// exact). The dot products as m16n8k16 bf16 MMAs with fp32 accumulation: the
// products are exact and their sums too while they fit 24 bits, as the
// official's bf16 einsum; the head sum in fp32 from +0, then bf16 (the
// official's torch sum; equal while exact). CAND (a Reindex layer): positions
// outside the candidate blocks (a byte per block of 8, 0 = out) score -inf,
// the official's masked_fill. Into scores [T][CAP] bf16 (the node's buffer;
// positions >= n not written). Then the node's counter t += 1 (a reader of
// token t waits for NCH).
// Per 256 positions of the chunk, warp w: positions 32 w .. 32 w + 31 (4 n8
// tiles), both 16-head halves (2 m16 tiles), 8 k-steps. The B fragments come
// straight from the staged fp4 rows (a register is one e2m1 byte: 2 adjacent
// columns).
#pragma once
#include "../core.cuh"
#include "act_quant.cuh"
#include "indexer_common.cuh"

namespace static_mk {

// timing build: the task's 2 stage stamps at step boundaries IS_STAMP0 and IS_STAMP1 (thread 0): 1 the inputs seen (the K row,
// the weights, q's count), 2 q and the K pages in shared memory, 3 the MMAs and the head sums done (scores stored), 4 the end
#ifndef IS_STAMP0
#define IS_STAMP0 1
#endif
#ifndef IS_STAMP1
#define IS_STAMP1 3
#endif
#define IS_MARK(p)                                                             \
  if ((p) == IS_STAMP0) {                                                      \
    STAGE_STAMP_T0(0)                                                          \
  }                                                                            \
  if ((p) == IS_STAMP1) {                                                      \
    STAGE_STAMP_T0(1)                                                          \
  }

constexpr int IS_QLD = IX_HD + 8; // q's row stride in shared memory (bf16)
constexpr int IS_OFF_Q = 0,       // q [32][IS_QLD] bf16
    IS_OFF_K = IS_OFF_Q + 32 * IS_QLD * 2; // the chunk's index K pages (raw)
// then w [32] fp32
__host__ __device__ constexpr int is_off_w(int ch) {
  return IS_OFF_K + ch * IX_ROW;
}
// a task's positions: 256 x the fewest halves that make NCH tasks of CAP
__host__ __device__ constexpr int is_ch(int cap, int nch) {
  return (cap / 256 + nch - 1) / nch * 256;
}

__device__ __forceinline__ void is_ldsm_x4(uint32_t addr, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, "
               "[%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void
    is_mma(float (&c)[4], uint32_t const (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, "
               "%2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

template <int H,
          int HD,
          int CAP,
          int R,
          bool CAND,
          int WP,
          int WN,
          bool KNODE,
          int NCH>
__device__ __forceinline__ void indexer_score_task(__nv_bfloat16 const *q,
                                                   float const *wpart,
                                                   unsigned char const *kc,
                                                   int const *pages,
                                                   unsigned char const *cand,
                                                   int const *pos,
                                                   __nv_bfloat16 *out,
                                                   uint32_t const *q_cnt,
                                                   uint32_t const *k_cnt,
                                                   uint32_t *done_cnt,
                                                   int t,
                                                   int c,
                                                   char *sm STAGE_STAMP_PARAM) {
  constexpr int CH = is_ch(CAP, NCH);
  static_assert(H == 32 && HD == IX_HD && CAP % 256 == 0 &&
                    (CAP / 256 + CH / 256 - 1) / (CH / 256) == NCH &&
                    IS_OFF_K % 16 == 0,
                "indexer_score: 32 heads of 128 columns, whole halves");
  int const tid = threadIdx.x, w = tid >> 5, l = tid & 31, g8 = l >> 2,
            qd = l & 3;
  int const p = pos[t], n = (p + 1) / R, k0 = c * CH;
  __syncthreads(); // the shared memory may be the ring's of a GEMM before
  if (k0 < n) {
    __nv_bfloat16 *const qs = reinterpret_cast<__nv_bfloat16 *>(sm + IS_OFF_Q);
    unsigned char const *const kr =
        reinterpret_cast<unsigned char const *>(sm + IS_OFF_K);
    float *const ws = reinterpret_cast<float *>(sm + is_off_w(CH));
    if constexpr (KNODE) { // index_k_post wrote token t's row this launch
      if (tid == 0) {
        cnt_wait(k_cnt + t, 1);
      }
      __syncthreads();
    }
    // the chunk's halves below n (inside CAP): their 2 pages each, whole, as
    // the framework lays them out (cp.async.cg: L2)
    int const nh =
        CH == 256 ? 1 : min(CH, min((n + 255) / 256 * 256, CAP) - k0) / 256;
    {
      uint32_t const dst = su32(sm + IS_OFF_K);
      constexpr int PG16 = IX_PAGE * IX_ROW / 16; // a page's 16-B chunks
      int const *const pt = pages + (size_t)t * (CAP / IX_PAGE) + k0 / IX_PAGE;
      for (int i = tid; i < nh * 2 * PG16; i += 256) {
        unsigned char const *src =
            kc + (size_t)pt[i / PG16] * IX_PAGE * IX_ROW + (i % PG16) * 16;
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst + 16 * i),
            "l"(src)
            : "memory");
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
    if (tid < H) { // w: weights_proj's K parts (polled) -> bf16 -> x 2^-6
      float v[WP];
      bool ok[WP];
#pragma unroll
      for (int k = 0; k < WP; k++) {
        ok[k] = false;
      }
      bool all;
      do {
        all = true;
#pragma unroll
        for (int k = 0; k < WP; k++) {
          if (!ok[k]) {
            float const *a = wpart + ((size_t)k * T + t) * WN + tid;
            asm volatile("ld.relaxed.gpu.global.f32 %0, [%1];"
                         : "=f"(v[k])
                         : "l"(a));
            ok[k] = validf(v[k]);
          }
          all = all && ok[k];
        }
        if (!all) {
          poll_pause(STATIC_POLL_NS);
        }
      } while (!all);
      float s = v[0];
#pragma unroll
      for (int k = 1; k < WP; k++) { // in part order
        s += v[k];
      }
      ws[tid] = bf16r(s) * 0x1p-6f;
    }
    if (tid == 0) {
      cnt_wait(q_cnt + t, 1); // indexer_q wrote token t's q
    }
    __syncthreads();
    IS_MARK(1)
    { // q: 32 x 128 bf16, 2 x 16 B per thread (relaxed: written this launch)
      unsigned char const *qb =
          reinterpret_cast<unsigned char const *>(q + (size_t)t * H * HD);
#pragma unroll
      for (int k = 0; k < 2; k++) {
        int const i = tid + 256 * k, r = i >> 4, c8 = (i & 15) * 8;
        uint4 const v = ld16_relaxed(qb + ((size_t)r * HD + c8) * 2);
        *reinterpret_cast<uint4 *>(qs + r * IS_QLD + c8) = v;
      }
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    IS_MARK(2)
    uint32_t const qa =
        su32(qs) + ((l & 15) * IS_QLD + (l >> 4) * 8) * 2; // rows 0..15
#pragma unroll 1
    for (int h = 0; h < CH / 256; h++) { // a half: 256 positions
      if (h > 0 && h >= nh) {
        break;
      }
      unsigned char const *const kh = kr + 256 * h * IX_ROW;
      int const k1 = k0 + 256 * h;
      float acc[2][4][4] = {};
#pragma unroll 2
      for (int ks = 0; ks < HD / 16; ks++) {
        uint32_t a0[4], a1[4];
        is_ldsm_x4(qa + ks * 32, a0);
        is_ldsm_x4(qa + 16 * IS_QLD * 2 + ks * 32, a1);
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
          int const pp = 32 * w + 8 * nt + g8, o = pp % IX_PAGE;
          unsigned char const *const pg = kh + (pp / IX_PAGE) * IX_PAGE * IX_ROW;
          unsigned char const *row = pg + o * (HD / 2);
          float const s = __uint_as_float(
              (uint32_t)pg[IX_PAGE * (HD / 2) + o * (HD / 32) + ks / 2] << 23);
          uint32_t const b0 = fp4_bf16x2(row[8 * ks + qd], s),
                         b1 = fp4_bf16x2(row[8 * ks + 4 + qd], s);
          is_mma(acc[0][nt], a0, b0, b1);
          is_mma(acc[1][nt], a1, b0, b1);
        }
      }
      // thread: heads g8, g8 + 8, 16 + g8, 24 + g8 of positions 2 qd, 2 qd + 1
      // of each n8 tile; the head sum over the 8 lanes of a column (xor 4, 8,
      // 16)
      float const w0 = ws[g8], w1 = ws[g8 + 8], w2 = ws[16 + g8],
                  w3 = ws[24 + g8];
      float sum[4][2];
#pragma unroll
      for (int nt = 0; nt < 4; nt++) {
#pragma unroll
        for (int e = 0; e < 2; e++) {
          float const s0 = fmaxf(bf16r(acc[0][nt][e]), 0.f),
                      s1 = fmaxf(bf16r(acc[0][nt][2 + e]), 0.f),
                      s2 = fmaxf(bf16r(acc[1][nt][e]), 0.f),
                      s3 = fmaxf(bf16r(acc[1][nt][2 + e]), 0.f);
          float u = 0.f;
          u = __fadd_rn(u, bf16r(__fmul_rn(s0, w0)));
          u = __fadd_rn(u, bf16r(__fmul_rn(s1, w1)));
          u = __fadd_rn(u, bf16r(__fmul_rn(s2, w2)));
          u = __fadd_rn(u, bf16r(__fmul_rn(s3, w3)));
          for (int d = 4; d < 32; d <<= 1) {
            u = __fadd_rn(u, __shfl_xor_sync(0xffffffffu, u, d));
          }
          sum[nt][e] = u;
        }
      }
      if (g8 == 0) {
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
          int const kk = k1 + 32 * w + 8 * nt + 2 * qd; // even: one block of 8
          if (kk < n) {
            __nv_bfloat162 v = __floats2bfloat162_rn(sum[nt][0], sum[nt][1]);
            if constexpr (CAND) {
              if (cand[(size_t)t * (CAP / 8) + kk / 8] == 0) {
                v = __floats2bfloat162_rn(-INFINITY, -INFINITY);
              }
            }
            *reinterpret_cast<__nv_bfloat162 *>(out + (size_t)t * CAP + kk) = v;
          }
        }
      }
    }
  }
  __syncthreads();
  if (k0 < n) {
    IS_MARK(3)
  }
  if constexpr (REARM) {
    // weights_proj's parts have several readers (token t's NCH tasks): the last
    // to count re-arms them (the others counted after their reads), once the
    // GEMM wrote them (a token with no position reads none)
    __shared__ uint32_t last_s;
    if (tid == 0) {
      uint32_t old; // acq_rel: the CTA's stores before (the barrier); the
                    // others' reads before the re-arm
      asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], 1;"
                   : "=r"(old)
                   : "l"(done_cnt + t)
                   : "memory");
      last_s = old == NCH - 1;
    }
    __syncthreads();
    if (last_s && tid < H) {
#pragma unroll 1
      for (int k = 0; k < WP; k++) {
        float *const a =
            const_cast<float *>(wpart) + ((size_t)k * T + t) * WN + tid;
        float v;
        do {
          asm volatile("ld.relaxed.gpu.global.f32 %0, [%1];"
                       : "=f"(v)
                       : "l"(a)
                       : "memory");
        } while (!validf(v));
        *reinterpret_cast<uint32_t *>(a) = 0xFFFFFFFFu;
      }
    }
  } else if (tid == 0) {
    cnt_add(done_cnt + t, 1); // release: the CTA's stores before (the barrier)
  }
}

#ifdef IS_TCGEN05
// ---- -DIS_TCGEN05: the dot products on the tcgen05 tensor core ----
// Per page g of the chunk (IX_PAGE = 128 positions) one product M 128 (the
// page's positions, the TMEM lanes) x N 32 (the heads, 32 fp32 TMEM columns) x
// K 128, as 2 instructions of K 64, kind::mxf4 with block scales: both
// operands packed e2m1 in shared memory, K-major with the 64-B swizzle (16-B
// chunk c of row r at chunk c ^ (r / 2 % 4)), their e8m0 scales per 32 columns
// in TMEM (4 columns a chunk of 128 rows; the scale ids of K step k: 2 k).
//   A  the page's e2m1 bytes as the cache holds them (64 B a position), each
//      16-B chunk copied to its swizzled place (cp.async), no conversion:
//      position o of the page is MMA row 32 (o % 4) + o / 4. Then the page's
//      scale bytes as the cache holds them (4 a position, 16-B chunk c:
//      positions 4 c .. 4 c + 3) are the scale chunk (row r's 4 bytes at 16
//      (r % 32) + 4 (r / 32); cp_sf, runtime.cuh).
//   B  q: indexer_q's bf16 values of a group of 32 are e2m1 values times one
//      power of 2; the task takes 2^e' with the group's largest magnitude in
//      [4 x 2^e', 8 x 2^e') (every value / 2^e' is then an e2m1 value: the
//      conversion is exact; e' >= -127) and stores the packed bytes ([32][64
//      B]) and the scale bytes e' + 127 (head h's 4 at 16 h of its chunk).
//
// Thread 0 copies the scale chunks to TMEM and issues the MMAs (per page 2,
// then a commit on the page's mbarrier). A thread reads the 32 heads of its
// TMEM lane 32 (warp % 4) + lane (warps 0..3: the half's page 0, 4..7: page
// 1), position 4 lane + warp % 4 of the page, and applies the bf16 path's
// arithmetic in its order: per g < 8 the heads g, g + 8, g + 16, g + 24 from
// +0, then ((0 + 1) + (2 + 3)) + ((4 + 5) + (6 + 7)). A chunk of more than 2
// halves: in rounds of 2 halves (4 pages), a CTA barrier between them.
//
// TMEM (the kernel's 512 columns; no MMA of another task is in flight between
// the task's first and last CTA barrier, and every GEMM overwrites its
// accumulators and scale columns before it reads them): page g < 4 of a
// round: the accumulators at IS5_ACC + 32 g, its scales at IS5_KSF + 4 g; q's
// scales at IS5_QSF: columns 64..211, which no GEMM of the ring uses (0..63
// accumulators, 232.. scales). The mbarriers are the task's own (is5_bars:
// static shared memory, never another task's), initialised by each task.
//
// The scores have the bf16 path's bits while the dot products are exact (they
// are while they fit 24 bits, in both paths) and no accumulator is a NaN (none
// is with finite q and K; a NaN would count as 0 in the bf16 path's fmaxf and
// stay a NaN here).
constexpr int IS5_PG = IX_PAGE * (IX_HD / 2);             // a page's e2m1 bytes
constexpr int IS5_ACC = 64, IS5_KSF = 192, IS5_QSF = 208; // TMEM columns
// shared memory: the chunk's pages [CH / 128][128][64 B], q [32][64 B], the
// pages' scale chunks, q's scale chunk, w [32] fp32
__host__ __device__ constexpr int is5_off_q(int ch) {
  return ch / IX_PAGE * IS5_PG;
}
__host__ __device__ constexpr int is5_off_ksf(int ch) {
  return is5_off_q(ch) + 32 * (IX_HD / 2);
}
__host__ __device__ constexpr int is5_off_qsf(int ch) {
  return is5_off_ksf(ch) + ch / IX_PAGE * SF_CHUNK;
}
__host__ __device__ constexpr int is5_off_w(int ch) {
  return is5_off_qsf(ch) + SF_CHUNK;
}
__shared__ __align__(8) uint64_t is5_bars[4]; // a page's MMAs are done

// kind::mxf4: e2m1 x e2m1 -> fp32, ue8m0 scales, both K-major, M 128, N 32;
// the scale id sf (0 or 2: the K step's first group of 32) of both operands
__device__ __forceinline__ uint32_t is5_idesc(uint32_t sf) {
  return (1u << 7) | (1u << 10) | ((32u / 8u) << 17) | (1u << 23) |
         ((128u / 16u) << 24) | (sf << 4) | (sf << 29);
}
// packed e2m1 rows of 64 B, K-major, 64-B swizzle (8 rows: 512 B)
__device__ __forceinline__ uint64_t is5_desc(uint32_t a) {
  return denc(a) | (denc(512) << 32ULL) | (1ULL << 46ULL) | (4ULL << 61ULL);
}
__device__ __forceinline__ void is5_mma(uint32_t d,
                                        uint64_t a,
                                        uint64_t b,
                                        uint32_t idesc,
                                        uint32_t en,
                                        uint32_t sfa,
                                        uint32_t sfb) {
  asm volatile("{\n.reg .pred p;\nsetp.ne.b32 p, %4, 0;\n"
#if (__CUDACC_VER_MAJOR__ > 12) ||                                             \
    (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 9)
               "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.block32 [%0], "
#else
               "tcgen05.mma.cta_group::1.kind::mxf4.block_scale.scale_vec::2X "
               "[%0], "
#endif
               "%1, %2, %3, [%5], [%6], p;\n}" ::"r"(d),
               "l"(a),
               "l"(b),
               "r"(idesc),
               "r"(en),
               "r"(sfa),
               "r"(sfb));
}
// 4 e2m1 values -> 2 bytes (a in byte 0's low nibble), exact
__device__ __forceinline__ uint32_t is5_pack4(float a,
                                              float b,
                                              float c,
                                              float d) {
  uint16_t r;
  asm("{\n.reg .b8 b0, b1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %2, %1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %4, %3;\n"
      "mov.b16 %0, {b0, b1};\n}"
      : "=h"(r)
      : "f"(a), "f"(b), "f"(c), "f"(d));
  return r;
}

// Two heads' terms: u += bf16(relu(bf16(acc)) x w), the bf16 path's
// __fadd_rn(u, bf16r(__fmul_rn(fmaxf(bf16r(acc), 0), w))) in fewer
// instructions: cvt.rn.relu.bf16x2.f32 gives the bits of bf16(fmaxf(bf16r(x),
// 0)) for every fp32 x that is not a NaN, and add.rn.f32.bf16 those of
// __fadd_rn(u, the bf16 value as fp32) (checked on a B200 against these
// expressions: all 2^32 x; 2^33 pairs). The product in fp32 with the fp32 w,
// as in the bf16 path.
__device__ __forceinline__ void
    is5_term2(float a0, float a1, float w0, float w1, float &u0, float &u1) {
  uint32_t p;
  asm("cvt.rn.relu.bf16x2.f32 %0, %1, %2;" : "=r"(p) : "f"(a1), "f"(a0));
  __nv_bfloat162 const t =
      __floats2bfloat162_rn(__fmul_rn(__uint_as_float(p << 16), w0),
                            __fmul_rn(__uint_as_float(p & 0xFFFF0000u), w1));
  asm("{\n.reg .b16 lo, hi;\nmov.b32 {lo, hi}, %2;\n"
      "add.rn.f32.bf16 %0, lo, %0;\nadd.rn.f32.bf16 %1, hi, %1;\n}"
      : "+f"(u0), "+f"(u1)
      : "r"(*reinterpret_cast<uint32_t const *>(&t)));
}

template <int H,
          int HD,
          int CAP,
          int R,
          bool CAND,
          int WP,
          int WN,
          bool KNODE,
          int NCH>
__device__ __forceinline__ void
    indexer_score_task5(__nv_bfloat16 const *q,
                        float const *wpart,
                        unsigned char const *kc,
                        int const *pages,
                        unsigned char const *cand,
                        int const *pos,
                        __nv_bfloat16 *out,
                        uint32_t const *q_cnt,
                        uint32_t const *k_cnt,
                        uint32_t *done_cnt,
                        int t,
                        int c,
                        char *sm,
                        uint32_t base,
                        uint32_t tb STAGE_STAMP_PARAM) {
  constexpr int CH = is_ch(CAP, NCH);
  static_assert(H == 32 && HD == IX_HD && CAP % 256 == 0 &&
                    (CAP / 256 + CH / 256 - 1) / (CH / 256) == NCH,
                "indexer_score: 32 heads of 128 columns, whole halves");
  static_assert(IX_PAGE == 128 && SF_CHUNK == 4 * IX_PAGE,
                "indexer_score (tcgen05): a page = the MMA's M, its scale "
                "words one scale chunk");
  int const tid = threadIdx.x, w = tid >> 5, l = tid & 31;
  int const p = pos[t], n = (p + 1) / R, k0 = c * CH;
  uint32_t const bar = su32(is5_bars);
  __syncthreads(); // the shared memory may be the ring's of a GEMM before
  if (k0 < n) {
    float *const ws = reinterpret_cast<float *>(sm + is5_off_w(CH));
    if constexpr (KNODE) { // index_k_post wrote token t's row this launch
      if (tid == 0) {
        cnt_wait(k_cnt + t, 1);
      }
      __syncthreads();
    }
    // the chunk's halves below n (inside CAP): their 2 pages each, whole
    // (cp.async.cg: L2): position o's 4 chunks of e2m1 to MMA row 32 (o % 4)
    // + o / 4 (swizzled), the 32 scale chunks as they are
    int const nh =
        CH == 256 ? 1 : min(CH, min((n + 255) / 256 * 256, CAP) - k0) / 256;
    {
      constexpr int PG16 = IX_PAGE * IX_ROW / 16, // a page's 16-B chunks
          KV16 = IS5_PG / 16;                     // of them, the e2m1 bytes'
      int const *const pt = pages + (size_t)t * (CAP / IX_PAGE) + k0 / IX_PAGE;
      for (int i = tid; i < nh * 2 * PG16; i += 256) {
        int const g = i / PG16, j = i % PG16, o = j >> 2,
                  rw = 32 * (o & 3) + (o >> 2);
        unsigned char const *src =
            kc + (size_t)pt[g] * IX_PAGE * IX_ROW + 16 * j;
        uint32_t const dst =
            j < KV16 ? base + g * IS5_PG + rw * 64 +
                           (((j & 3) ^ ((rw >> 1) & 3)) << 4)
                     : base + is5_off_ksf(CH) + g * SF_CHUNK + 16 * (j - KV16);
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst),
                     "l"(src)
                     : "memory");
      }
      asm volatile("cp.async.commit_group;" ::: "memory");
    }
    if (tid == 0) { // a page's MMAs done: one commit
#pragma unroll
      for (int g = 0; g < 4; g++) {
        mbar_init(bar + 8 * g, 1);
      }
      asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    if (tid < H) { // w: weights_proj's K parts (polled) -> bf16 -> x 2^-6
      float v[WP];
      bool ok[WP];
#pragma unroll
      for (int k = 0; k < WP; k++) {
        ok[k] = false;
      }
      bool all;
      do {
        all = true;
#pragma unroll
        for (int k = 0; k < WP; k++) {
          if (!ok[k]) {
            float const *a = wpart + ((size_t)k * T + t) * WN + tid;
            asm volatile("ld.relaxed.gpu.global.f32 %0, [%1];"
                         : "=f"(v[k])
                         : "l"(a));
            ok[k] = validf(v[k]);
          }
          all = all && ok[k];
        }
        if (!all) {
          poll_pause(STATIC_POLL_NS);
        }
      } while (!all);
      float s = v[0];
#pragma unroll
      for (int k = 1; k < WP; k++) { // in part order
        s += v[k];
      }
      ws[tid] = bf16r(s) * 0x1p-6f;
    }
    if (tid == 0) {
      cnt_wait(q_cnt + t, 1); // indexer_q wrote token t's q
    }
    __syncthreads();
    IS_MARK(1)
    // q: head tid / 8, columns 16 (tid % 8) .. + 15 (2 threads a group of 32;
    // relaxed: written this launch) -> 8 bytes of e2m1, the group's scale byte
    int const hq = tid >> 3, c8 = tid & 7;
    unsigned char const *qb = reinterpret_cast<unsigned char const *>(
        q + ((size_t)t * H + hq) * HD + 16 * c8);
    uint4 const v0 = ld16_relaxed(qb), v1 = ld16_relaxed(qb + 16);
    uint32_t const wd[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
    float x[16], amax = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      float2 const f =
          __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const *>(&wd[k]));
      x[2 * k] = f.x;
      x[2 * k + 1] = f.y;
      amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
    }
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1)); // the group's
    // the scale byte e' + 127 = amax's exponent field - 2 (at least 0)
    uint32_t const ef = (__float_as_uint(amax) >> 23) & 0xFFu,
                   sb = ef > 2u ? ef - 2u : 0u;
    float const inv = __uint_as_float((254u - sb) << 23); // 2^-e'
#pragma unroll
    for (int k = 0; k < 16; k++) {
      x[k] = __fmul_rn(x[k], inv);
    }
    uint32_t const p0 = is5_pack4(x[0], x[1], x[2], x[3]) |
                        is5_pack4(x[4], x[5], x[6], x[7]) << 16,
                   p1 = is5_pack4(x[8], x[9], x[10], x[11]) |
                        is5_pack4(x[12], x[13], x[14], x[15]) << 16;
    asm volatile("st.shared.v2.b32 [%0], {%1, %2};" ::"r"(
                     base + is5_off_q(CH) + hq * 64 +
                     (((c8 >> 1) ^ ((hq >> 1) & 3)) << 4) + 8 * (c8 & 1)),
                 "r"(p0),
                 "r"(p1)
                 : "memory");
    // the head's 4 scale bytes (its 8 lanes: groups c8 / 2) -> line hq
    uint32_t const s2 = sb | __shfl_xor_sync(0xffffffffu, sb, 2) << 8,
                   s4 = s2 | __shfl_xor_sync(0xffffffffu, s2, 4) << 16;
    if (c8 == 0) {
      asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};" ::"r"(
                       base + is5_off_qsf(CH) + 16 * hq),
                   "r"(s4),
                   "r"(0u),
                   "r"(0u),
                   "r"(0u)
                   : "memory");
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    // the tensor core (the async proxy) reads q, the pages and the scales
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    __syncthreads();
    IS_MARK(2)
    // rounds of 2 halves (4 pages: the TMEM columns and the mbarriers of a
    // round; CH <= 512: one round). A later round starts when every thread
    // has read the one before (the CTA barrier), and a page's mbarrier
    // completes one phase a round
#pragma unroll 1
    for (int r = 0; r < (CH / 256 + 1) / 2; r++) {
      if (r > 0) {
        if (2 * r >= nh) {
          break;
        }
        fence_before(); // the round before's TMEM reads, before these MMAs
        __syncthreads();
      }
      if (tid == 0) { // the scale chunks -> TMEM, per page 2 MMAs, a commit
        fence_after();
        uint64_t const dq = is5_desc(base + is5_off_q(CH));
        uint32_t const sfb = tb + IS5_QSF;
        if (r == 0) {
          cp_sf(sfb, base + is5_off_qsf(CH));
        }
        int const np = min(4, 2 * nh - 4 * r); // the round's pages
#pragma unroll 1
        for (int g = 0; g < np; g++) {
          uint32_t const sfa = tb + IS5_KSF + 4 * g,
                         acc = tb + IS5_ACC + 32 * g;
          cp_sf(sfa, base + is5_off_ksf(CH) + (4 * r + g) * SF_CHUNK);
          uint64_t const dk = is5_desc(base + (4 * r + g) * IS5_PG);
          is5_mma(acc, dk, dq, is5_idesc(0), 0u, sfa, sfb);
          is5_mma(acc, dk + 2, dq + 2, is5_idesc(2), 1u, sfa, sfb); // + 32 B
          tc_commit(bar + 8 * g);
        }
      }
      __syncwarp(); // warp 0 waits below as one
#pragma unroll 1
      for (int hh = 0; hh < 2; hh++) { // a half: page hh 2 + w / 4, the row
                                       // of this thread's TMEM lane
        int const h = 2 * r + hh;
        if (hh > 0 && h >= nh) {
          break;
        }
        int const g = 2 * hh + (w >> 2),
                  kk = k0 + 256 * h + 128 * (w >> 2) + 4 * l + (w & 3);
        // the bf16 path stores the positions in pairs, a pair when its even
        // position is below n
        bool const st = (kk & ~1) < n;
        bool in = true; // its block of 8 is a candidate (loaded ahead)
        if constexpr (CAND) {
          if (st) {
            in = cand[(size_t)t * (CAP / 8) + kk / 8] != 0;
          }
        }
        mbar_wait(bar + 8 * g, r & 1);
        fence_after();
        // the head sum in the bf16 path's order: u[g8] = the heads g8, g8 + 8,
        // 16 + g8, 24 + g8 from +0 (the bf16 path's thread), then its butterfly
        // over g8 (xor 1, 2, 4). 8 heads a round of the loop (one copy of the
        // code)
        uint32_t const acol =
            tb + IS5_ACC + 32 * g + ((uint32_t)(32 * (w & 3)) << 16);
        float u[8];
#pragma unroll
        for (int g8 = 0; g8 < 8; g8++) {
          u[g8] = 0.f;
        }
#pragma unroll 1
        for (int j = 0; j < 4; j++) {
          float v[8];
          tmem_ld8(acol + 8 * j, v);
          float const *const wj = ws + 8 * j;
#pragma unroll
          for (int i = 0; i < 4; i++) {
            is5_term2(v[2 * i],
                      v[2 * i + 1],
                      wj[2 * i],
                      wj[2 * i + 1],
                      u[2 * i],
                      u[2 * i + 1]);
          }
        }
        float const sum =
            __fadd_rn(__fadd_rn(__fadd_rn(u[0], u[1]), __fadd_rn(u[2], u[3])),
                      __fadd_rn(__fadd_rn(u[4], u[5]), __fadd_rn(u[6], u[7])));
        if (st) {
          out[(size_t)t * CAP + kk] = __float2bfloat16_rn(in ? sum : -INFINITY);
        }
      }
    }
    fence_before(); // the TMEM reads, before the MMAs of a later task
    // the ring area's next TMA writes come after these generic stores
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  }
  __syncthreads();
  if (k0 < n) {
    IS_MARK(3)
  }
  if constexpr (REARM) { // as indexer_score_task
    __shared__ uint32_t last_s;
    if (tid == 0) {
      uint32_t old;
      asm volatile("atom.add.acq_rel.gpu.global.u32 %0, [%1], 1;"
                   : "=r"(old)
                   : "l"(done_cnt + t)
                   : "memory");
      last_s = old == NCH - 1;
    }
    __syncthreads();
    if (last_s && tid < H) {
#pragma unroll 1
      for (int k = 0; k < WP; k++) {
        float *const a =
            const_cast<float *>(wpart) + ((size_t)k * T + t) * WN + tid;
        float v;
        do {
          asm volatile("ld.relaxed.gpu.global.f32 %0, [%1];"
                       : "=f"(v)
                       : "l"(a)
                       : "memory");
        } while (!validf(v));
        *reinterpret_cast<uint32_t *>(a) = 0xFFFFFFFFu;
      }
    }
  } else if (tid == 0) {
    cnt_add(done_cnt + t, 1); // release: the CTA's stores before (the barrier)
  }
}
#endif // IS_TCGEN05

// indexer_score task (t, c), grid (T, NCH): chunk c of is_ch(CAP, NCH); QI:
// indexer_q's node (its buffer q_idx, a counter per token); WP: weights_proj's
// node (a gemm_tile in slots: WP::y K parts of WP::v[2] rows, the first H
// used); KC: the index K cache's producer (index_k_post, a counter per token;
// or a graph input: none to wait for). PARAMS {H, HD, CAP, R, CAND}; BUF_SLOTS
// {the index K cache
// (pages, indexer_common.cuh), positions ([T] int32), its page table ([T][CAP /
// IX_PAGE] int32), with CAND the candidate
// blocks ([T][CAP / 8] uint8)}, graph tensors; the node's buffer: scores
// [T][CAP] bf16, a counter per token
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class QI,
          class WP,
          class KC,
          class... REST>
__device__ __forceinline__ void run_indexer_score(Maps const &,
                                                  G const &g,
                                                  KernelLocals &L,
                                                  StaticTask const &tk) {
  constexpr int H = PARAMS::v[0], HD = PARAMS::v[1], CAP = PARAMS::v[2],
                R = PARAMS::v[3];
  constexpr bool CAND = PARAMS::v[4] != 0, KNODE = KC::counter >= 0;
  static_assert(SELF::x == T && SELF::z == 1,
                "indexer_score: grid (T, chunks of 256 x halves)");
  static_assert(SELF::buf >= 0 && SELF::counter >= 0,
                "indexer_score: its output in its buffer, a counter per token");
  static_assert(QI::buf >= 0 && QI::counter >= 0 && QI::v[0] == H &&
                    QI::v[1] == HD && QI::x == T,
                "indexer_score: q is indexer_q's output (a counter per token)");
  static_assert(WP::buf >= 0 && WP::v[0] == COMBINE_SLOTS && WP::v[2] >= H &&
                    WP::y >= 1 && WP::y <= 8,
                "indexer_score: weights_proj's K parts in slots (at most 8)");
  static_assert(!KNODE || (KC::x == T && KC::y == 1),
                "indexer_score: the index K cache is index_k_post's (a "
                "counter per token) or a graph input");
  unsigned char const *cand = nullptr;
  if constexpr (CAND) {
    cand = buf_at<unsigned char const>(g, slot_at<BUF_SLOTS, 3>());
  }
#ifdef IS_TCGEN05
  indexer_score_task5<H, HD, CAP, R, CAND, WP::y, WP::v[2], KNODE, SELF::y>(
      buf_at<__nv_bfloat16 const>(g, QI::buf),
      buf_at<float const>(g, WP::buf),
      buf_at<unsigned char const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 2>()),
      cand,
      buf_at<int const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<__nv_bfloat16>(g, SELF::buf),
      g.cnt + QI::counter,
      g.cnt + (KNODE ? KC::counter : 0),
      g.cnt + SELF::counter,
      tk.x,
      tk.y,
      L.sm,
      L.base,
      L.tb STAGE_STAMP_ARG(L.stage_stamps));
#else
  indexer_score_task<H, HD, CAP, R, CAND, WP::y, WP::v[2], KNODE, SELF::y>(
      buf_at<__nv_bfloat16 const>(g, QI::buf),
      buf_at<float const>(g, WP::buf),
      buf_at<unsigned char const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 2>()),
      cand,
      buf_at<int const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<__nv_bfloat16>(g, SELF::buf),
      g.cnt + QI::counter,
      g.cnt + (KNODE ? KC::counter : 0),
      g.cnt + SELF::counter,
      tk.x,
      tk.y,
      L.sm STAGE_STAMP_ARG(L.stage_stamps));
#endif
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_indexer_score() {
#ifdef IS_TCGEN05
  return is5_off_w(is_ch(PARAMS::v[2], SELF::y)) + 32 * 4;
#else
  return is_off_w(is_ch(PARAMS::v[2], SELF::y)) + 32 * 4;
#endif
}

} // namespace static_mk
