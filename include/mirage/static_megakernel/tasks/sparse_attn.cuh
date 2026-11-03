// tasks/sparse_attn.cuh -- DeepSeek-V4.1's attention of a sliding-window layer
// (compress_ratio 0: the window only) in a decode step, through the act_quant
// of o before wo_a (model.py L781-787): task (t, g) of grid (T, H / 16), token
// t, heads [16 g, 16 g + 16). The official's sparse_attn (kernel.py
// sparse_attn_kernel, as tilelang compiles it: mma.sync m16n8k16, expf, no
// fast math) is redone step for step, so the same inputs give the same bits:
//   - the keys in the official's list order (get_window_topk_idxs): list
//     position i = slot (p % W + 1 + i) % W, the oldest first; slots > p (not
//     yet written, -1 in the list) give S = -inf and a K row of 0
//   - S = q K^T: bf16 x bf16 MMAs over the 512 columns in order, from 0; then
//     x fp32(512^-0.5)
//   - blocks of 64 list positions (2 for W = 128): per head the running max m
//     (from -1e30), P = expf(S - m), the block's sum of P in the official's
//     order (per lane 8 values, then the 2 key halves, then lanes ^ 2, ^ 1),
//     sum = sum x expf(m_prev - m) + the block's (fma), O = O x expf(m_prev -
//     m) + bf16(P) V (V = K, the MMAs over the block's 64 positions in order)
//   - O / (sum + expf(sink - m)) -> bf16 (o_attn), the inverse RoPE of the last
//     RD = 64 columns at p (torch's complex multiply by conj(freqs_cis)) ->
//     bf16 (o), act_quant's formula per 32 columns -> o_q
// K comes from the window cache [T][W][D + D / 32] (kv_post.cuh: e4m3 bytes,
// then e8m0 scales), dequantized to bf16 (exact, as the official's cache). A
// Reuse layer's compressed K: dequantized once per token by the node's tasks
// while q is on its way, into a bf16 buffer the pairs copy from (sa_bf16k).
// Split (grid z = NS = 1 + NC / 128, the compressed layers with fewer tasks
// per token than pairs: H < 64): task (t, g, s) the list's pair s alone, its K
// dequantized before q is needed. After S its 2 block maxima per head go to
// the others (stored into a buffer of 0xFF bytes before each launch, polled by
// the later pairs' tasks: no flag), and its softmax starts from the official's
// running max (that of the earlier pairs' blocks):
// its P, rescales and block sums are the official's bits. Its O (from 0) and
// block sums -> the partials, the group's tasks counted in counter 1 + t H /
// 16 + g; then each combines its 1 / NS of the group's columns, replaying the
// official's chain (sum: the same fma's, bit for bit; O = O sc0 sc1 + O_pair:
// the official adds P V into O in its MMAs, so O's last bits may differ) and
// runs the tail on them (the maxima first: with each pair's own max, the
// rescales would round differently from the official's)
// Warps: S, warp w list positions 16 w .. 16 w + 15; softmax, thread (head r,
// block b, key half, lane quarter) as the official's lanes; P V, warp w columns
// 64 w .. 64 w + 63 (warp 7 the RoPE columns). The output, the node's buffer
// laid out as act_quant's (o_q [T][H D], then the scale chunks; every byte
// written each launch); then the node's counter += 1 (wo_a waits for all
// T H / 16 tasks).
#pragma once
#include "../core.cuh"
#include "act_quant.cuh"

namespace static_mk {

// shared memory (dynamic, from the base: in the ring area, where no GEMM load
// is in flight between tasks)
constexpr int SA_HPT = 16;      // heads per task: the MMA's M
constexpr int SA_QLD = 512 + 8; // bf16 row stride of q and K (1040 B: an
                                // ldmatrix's 8 rows on distinct banks)
constexpr int SA_PLD = 128 + 8; // P's bf16 row stride
constexpr int SA_SLD = 128 + 4; // S's fp32 row stride
constexpr int SA_OFF_Q = 0, // q [16][512] bf16; after S, o_q's bytes [16][512]
    SA_OFF_K = SA_OFF_Q + SA_HPT * SA_QLD * 2, // K [128][512] bf16, by list
                                               // position
    SA_OFF_P = SA_OFF_K + 128 * SA_QLD * 2,    // P [16][128] bf16
    SA_OFF_S = SA_OFF_P + SA_HPT * SA_PLD * 2, // S [16][128] fp32
    SA_OFF_R = SA_OFF_S + SA_HPT * SA_SLD * 4, // per head: the pair's 2
                                               // rescales, the denominator
    SA_OFF_E = SA_OFF_R + SA_HPT * 4 * 4,      // o's scale bytes [16][16]
    SA_OFF_F = SA_OFF_E + SA_HPT * 16,         // freqs at p: [RD / 2] float2
    SA_OFF_O =
        SA_OFF_F + 32 * 8;       // O [16][SA_OSLD] fp32 for the tail (its own
                                 // place: the dry pass's tail must not touch K)
constexpr int SA_OSLD = 512 + 8; // O's row stride (8 banks a row: no conflicts)
constexpr int SA_CROW = 288;     // a compressed cache row: 512 e2m1 (low nibble
                                 // first), then 32 e4m3 scales (one per 16)
// the staging area of a pair's 128 compressed rows (raw bytes, cp.async) in
// O's place (O is written only in the tail, after the last pair), then the
// compressed list's ids [512], the dry pass's dequant scratch (2 rows)
constexpr int SA_OFF_X = SA_OFF_O,
              SA_OFF_I = SA_OFF_O + (SA_HPT * SA_OSLD * 4 > 128 * SA_CROW
                                         ? SA_HPT * SA_OSLD * 4
                                         : 128 * SA_CROW),
              SA_OFF_Y = SA_OFF_I + 512 * 4,
              SA_OFF_B = SA_OFF_Y + 2 * 512 * 2, // 2 mbarriers: the halves of a
                                                 // pair's bf16 K (sa_bf16k)
    SA_SMEM = SA_OFF_B + 16;
static_assert(SA_SMEM % 16 == 0 && SA_SMEM <= 220 * 1024,
              "sparse_attn: its shared memory from the base: the ring area "
              "and past it (no GEMM runs beside it on its SM; a GEMM refills "
              "every slot it reads)");

__device__ __forceinline__ void sa_ldsm_x4(uint32_t addr, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, "
               "[%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void sa_ldsm_x4_t(uint32_t addr, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, "
               "%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
// c += a b: bf16 x bf16 -> fp32, m16n8k16 (the instruction tilelang's
// T.gemm uses here)
__device__ __forceinline__ void
    sa_mma(float (&c)[4], uint32_t const (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, "
               "%2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// 16 e4m3 values (v, in column order) times 2^(s - 127) (s: their e8m0 scale
// byte) as bf16 at dst: exact (an e4m3 value times a power of 2 has at most 4
// significant bits)
__device__ __forceinline__ void
    sa_dequant16(uint4 v, uint32_t s, __nv_bfloat16 *dst) {
  float const sc = __uint_as_float(s << 23);
  uint32_t const wd[4] = {v.x, v.y, v.z, v.w};
  uint32_t o[8];
#pragma unroll
  for (int k = 0; k < 8; k++) {
    uint16_t const two = (uint16_t)(wd[k >> 1] >> (16 * (k & 1)));
    uint32_t h;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"(two));
    float2 const f = __half22float2(*reinterpret_cast<__half2 const *>(&h));
    __nv_bfloat162 const b = __floats2bfloat162_rn(f.x * sc, f.y * sc);
    o[k] = *reinterpret_cast<uint32_t const *>(&b);
  }
  reinterpret_cast<uint4 *>(dst)[0] = make_uint4(o[0], o[1], o[2], o[3]);
  reinterpret_cast<uint4 *>(dst)[1] = make_uint4(o[4], o[5], o[6], o[7]);
}

// q for this task's 16 heads, from wq_b's QP K
// parts in slots: polled (all loads in flight), added in part order -> bf16
// (wq_b's output) -> RoPE of each head's last RD columns at position p (re =
// fma(a, c, -bd), im = fma(b, c, ad)) -> bf16 -> qs. Thread: float4 i = tid +
// 256 k (k < 8) of the 16 x 512 values: head i / 128, columns 4 (i % 128) ..
// + 3. real = false (the dry pass): one round of loads, zeros stored. RA: this
// task re-arms what it read (REARM); false: the node's split tasks all read
// them, and each re-arms its share after the exchange (sa_split)
template <int H, int D, int RD, int QP, bool RA = true>
__device__ __forceinline__ void sa_q_parts(float const *part,
                                           float const *freqs,
                                           __nv_bfloat16 *qs,
                                           int t,
                                           int g,
                                           int p,
                                           bool real) {
  constexpr size_t N = (size_t)H * D;
  float const *src = part + (size_t)t * N + (size_t)g * SA_HPT * D;
  float4 v[QP][8];
  bool ok[QP];
#pragma unroll
  for (int pp = 0; pp < QP; pp++) {
    ok[pp] = false;
  }
  bool all;
  do {
    all = true;
#pragma unroll
    for (int pp = 0; pp < QP; pp++) {
      if (!ok[pp]) {
#pragma unroll
        for (int k = 0; k < 8; k++) {
          v[pp][k] =
              ldf4_poll(src + (size_t)pp * T * N + 4 * (threadIdx.x + 256 * k));
        }
      }
    }
#pragma unroll
    for (int pp = 0; pp < QP; pp++) {
      if (!ok[pp]) {
        bool o = true;
#pragma unroll
        for (int k = 0; k < 8; k++) {
          o = o && validf4(v[pp][k]);
        }
        ok[pp] = o;
      }
      all = all && ok[pp];
    }
    if (!all && real) {
      poll_pause(STATIC_POLL_NS);
    }
  } while (!all && real);
#pragma unroll
  for (int k = 0; k < 8; k++) {
    int const i = threadIdx.x + 256 * k, r = i >> 7, c = 4 * (i & 127);
    float x[4] = {v[0][k].x, v[0][k].y, v[0][k].z, v[0][k].w};
#pragma unroll
    for (int pp = 1; pp < QP; pp++) { // in part order
      x[0] += v[pp][k].x;
      x[1] += v[pp][k].y;
      x[2] += v[pp][k].z;
      x[3] += v[pp][k].w;
    }
#pragma unroll
    for (int e = 0; e < 4; e++) {
      x[e] = bf16r(x[e]); // wq_b's output
    }
    if (c >= D - RD) { // 2 RoPE pairs: (c, c + 1), (c + 2, c + 3)
      float4 const f = *reinterpret_cast<float4 const *>(
          freqs + ((size_t)p * (RD / 2) + (c - (D - RD)) / 2) * 2);
      float const a0 = x[0], b0 = x[1], a1 = x[2], b1 = x[3];
      x[0] = __fmaf_rn(a0, f.x, -__fmul_rn(b0, f.y));
      x[1] = __fmaf_rn(b0, f.x, __fmul_rn(a0, f.y));
      x[2] = __fmaf_rn(a1, f.z, -__fmul_rn(b1, f.w));
      x[3] = __fmaf_rn(b1, f.z, __fmul_rn(a1, f.w));
    }
    __nv_bfloat162 const h0 = __floats2bfloat162_rn(real ? x[0] : 0.f,
                                                    real ? x[1] : 0.f),
                         h1 = __floats2bfloat162_rn(real ? x[2] : 0.f,
                                                    real ? x[3] : 0.f);
    *reinterpret_cast<uint2 *>(qs + r * SA_QLD + c) =
        make_uint2(*reinterpret_cast<uint32_t const *>(&h0),
                   *reinterpret_cast<uint32_t const *>(&h1));
    if constexpr (REARM && RA) { // the one reader of these parts (config.cuh)
      if (real) {
#pragma unroll
        for (int pp = 0; pp < QP; pp++) {
          rearm16(src + (size_t)pp * T * N + 4 * (threadIdx.x + 256 * k));
        }
      }
    }
  }
}

// timing build: the task's 2 stage stamps at step boundaries SA_STAMP0 and
// SA_STAMP1 (thread 0, after the barrier ending the step): 0 K rows of earlier
// launches dequantized, 1 inputs ready (the waits), 2 the new row and q in
// shared memory, 3 S, 4 softmax, 5 O staged as o_q, 6 o_q stored, 7 the dry
// pass's end, 8 P V done (thread 0's warp), 9 the tail done (thread 0), 11 /
// 12 / 13 / 15 / 14 pair 1's S / softmax / P V / staging waited / next K
// dequantized, 10
// q from wq_b's parts done (thread 0, the real pass) (2..5, 8, 9: the real
// pass); split: 16 the partial and sums stored, 18 the group's sums and shares
// polled; a Full layer's compressed pair: 19 its list and new row
// counted, 20 its rows staged
#ifndef SA_STAMP0
#define SA_STAMP0 1
#endif
#ifndef SA_STAMP1
#define SA_STAMP1 5
#endif
#define SA_MARK(p)                                                             \
  if (threadIdx.x == 0) {                                                      \
    if ((p) == SA_STAMP0) {                                                    \
      STAGE_STAMP_AT(0)                                                        \
    }                                                                          \
    if ((p) == SA_STAMP1) {                                                    \
      STAGE_STAMP_AT(1)                                                        \
    }                                                                          \
  }

// a pair's 128 compressed rows (list positions 128 pr .. + 127 of ids), raw
// bytes into the staging area: its 2304 chunks of 16 B (row c / 18, bytes
// 16 (c % 18) ..) in order, thread chunks tid + 256 k (k < 9): a warp's
// cp.async 512 contiguous bytes of 2 or 3 rows. id -1:
// zeros (src-size 0: a row of 0, the official's, with no masking in the
// dequant). Dry pass (!real): row 0 for every row
template <int CAP>
__device__ __forceinline__ void sa_stage_rows(unsigned char const *ccache,
                                              int const *ids,
                                              int t,
                                              int pr,
                                              bool real,
                                              char *sm) {
  uint32_t const x = su32(sm + SA_OFF_X);
#pragma unroll
  for (int k = 0; k < 9; k++) {
    int const c = threadIdx.x + 256 * k, row = c / 18, o = 16 * (c - 18 * row);
    int const id = real ? ids[128 * pr + row] : 0;
    unsigned char const *src =
        ccache + ((size_t)t * CAP + (id >= 0 ? id : 0)) * SA_CROW + o;
    asm volatile(
        "cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(x + 16 * c),
        "l"(src),
        "r"(id >= 0 ? 16 : 0)
        : "memory");
  }
  asm volatile("cp.async.commit_group;" ::: "memory");
}
// 4 bytes of e2m1 x2 (low nibble first) -> their 8 values x 2^-112 as 4 bf16
// x2 (exact): f16 x2 (the bytes as .b8 operands: no shifts and masks to
// pick them), then each half's bits >> 3 (f16's 5 exponent bits into bf16's
// 8: the value x 2^(15 - 127); 0 stays 0)
__device__ __forceinline__ void sa_e2m1x8_bf16(uint32_t d, uint32_t (&o)[4]) {
  uint32_t h[4];
  asm("{\n.reg .b8 b0, b1, b2, b3;\nmov.b32 {b0, b1, b2, b3}, %4;\n"
      "cvt.rn.f16x2.e2m1x2 %0, b0;\ncvt.rn.f16x2.e2m1x2 %1, b1;\n"
      "cvt.rn.f16x2.e2m1x2 %2, b2;\ncvt.rn.f16x2.e2m1x2 %3, b3;\n}"
      : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
      : "r"(d));
#pragma unroll
  for (int k = 0; k < 4; k++) {
    o[k] = ((h[k] >> 3) & 0x0FFF0FFFu) | (h[k] & 0x80008000u);
  }
}
// the staged pair's rows -> bf16 K rows in ks: warp w rows 16 w .. + 15, 4 at
// a time (their loads first: 2 warps a scheduler leave the loads' latency
// unhidden); lane l columns 8 l .. + 7 and 256 + 8 l .. + 7 (a warp's loads
// and stores contiguous: no bank conflicts). A value: e2m1 x 2^-112 (above) x
// its group's e4m3 scale x 2^112, one bf16 multiply, exact (at most 6
// significant bits). id -1: staged as zeros. Dry pass: into a scratch row. gk:
// into these rows of global memory instead (the bf16 K buffer, rows SA_QLD
// apart: sa_bf16k)
__device__ __forceinline__ void
    sa_dequant_rows(bool real, char *sm, __nv_bfloat16 *gk = nullptr) {
  int const w = threadIdx.x >> 5, l = threadIdx.x & 31;
  unsigned char const *const xs =
      reinterpret_cast<unsigned char const *>(sm + SA_OFF_X);
#pragma unroll 1
  for (int r0 = 16 * w; r0 < 16 * w + 16; r0 += 4) {
    uint32_t d[4][2], sb[4];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      unsigned char const *const src = xs + (r0 + j) * SA_CROW;
      d[j][0] = *reinterpret_cast<uint32_t const *>(src + 4 * l);
      d[j][1] = *reinterpret_cast<uint32_t const *>(src + 128 + 4 * l);
      sb[j] = src[256 + (l >> 1)] | (uint32_t)src[272 + (l >> 1)] << 8;
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
      uint32_t sh;
      asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(sh) : "h"((uint16_t)sb[j]));
      float2 const f = __half22float2(*reinterpret_cast<__half2 const *>(&sh));
      __nv_bfloat162 const sc[2] = {__float2bfloat162_rn(f.x * 0x1p112f),
                                    __float2bfloat162_rn(f.y * 0x1p112f)};
      uint32_t ov[2][4];
#pragma unroll
      for (int c = 0; c < 2; c++) {
        sa_e2m1x8_bf16(d[j][c], ov[c]);
#pragma unroll
        for (int k = 0; k < 4; k++) {
          __nv_bfloat162 const b2 = __hmul2(
              *reinterpret_cast<__nv_bfloat162 const *>(&ov[c][k]), sc[c]);
          ov[c][k] = *reinterpret_cast<uint32_t const *>(&b2);
        }
      }
      __nv_bfloat16 *const dst =
          gk     ? gk + (r0 + j) * SA_QLD + 8 * l
          : real ? reinterpret_cast<__nv_bfloat16 *>(sm + SA_OFF_K) +
                       (r0 + j) * SA_QLD + 8 * l
                 : reinterpret_cast<__nv_bfloat16 *>(sm + SA_OFF_Y) + 8 * l;
      *reinterpret_cast<uint4 *>(dst) =
          make_uint4(ov[0][0], ov[0][1], ov[0][2], ov[0][3]);
      *reinterpret_cast<uint4 *>(dst + 256) =
          make_uint4(ov[1][0], ov[1][1], ov[1][2], ov[1][3]);
    }
  }
}

// whether the compressed K is dequantized once per token into the bf16 buffer:
// a Reuse layer (no node writes the list or the rows this launch) with a task
// per pair or more (H / 16 >= NC / 128: one pair each in the prologue);
// static_megakernel.py sparse_attn_layer: the same test gives the node the
// buffer, the pairs' counters and concurrent_along_axis 1. (With fewer tasks,
// 2 or 4 pairs in a task's prologue would outlast its wait for q.)
template <int NC, bool CNODE, bool CCNODE, int H>
__host__ __device__ constexpr bool sa_bf16k() {
  return NC > 0 && !CNODE && !CCNODE && H / SA_HPT >= NC / 128;
}

// sa_bf16k: half hf of compressed pair c of token t's list (kbf rows 128 c +
// 64 hf .., HALF bytes) -> K rows 64 hf .. of ks, one bulk copy completing on
// mbarrier hf (thread 0, after a barrier ending every warp's reads of those
// rows). Half 0 first waits for the pair's flag (the task that dequantized it)
// and orders the flag's acquire before the async proxy's reads
template <int NC, int NP, int HALF>
__device__ __forceinline__ void sa_copy_half(__nv_bfloat16 const *kbf,
                                             uint32_t const *done_cnt,
                                             int t,
                                             int c,
                                             int hf,
                                             char *sm) {
  uint32_t const mb = su32(sm + SA_OFF_B) + 8 * hf;
  if (hf == 0) {
    cnt_wait(done_cnt + 1 + NP * t + c, 1);
    asm volatile("fence.proxy.async.global;" ::: "memory");
  }
  mbar_expect(mb, HALF);
  asm volatile(
      "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::"
      "bytes [%0], [%1], %2, [%3];" ::"r"(su32(sm + SA_OFF_K) + hf * HALF),
      "l"(kbf + ((size_t)t * NC + 128 * c + 64 * hf) * SA_QLD),
      "r"(HALF),
      "r"(mb)
      : "memory");
}

// q: wq_b's QP K parts in slots (fp32 [QP][T][H D], 0xFF before the launch:
// polled), made into the task's bf16 q here (sa_q_parts)
template <int H,
          int D,
          int W,
          int RD,
          int QP,
          int SCALE_BITS,
          int NC,
          int CAP,
          bool CNODE,
          bool CCNODE,
          int NS>
__device__ __forceinline__ void sparse_attn_task(float const *q,
                                                 unsigned char const *cache,
                                                 int const *pos,
                                                 float const *freqs,
                                                 float const *sink,
                                                 unsigned char const *ccache,
                                                 int const *cidx,
                                                 __nv_bfloat16 *kbf,
                                                 float *part,
                                                 float *mlb,
                                                 unsigned char *out,
                                                 uint32_t const *kv_cnt,
                                                 uint32_t const *ci_cnt,
                                                 uint32_t const *cc_cnt,
                                                 uint32_t *done_cnt,
                                                 int t,
                                                 int g,
                                                 int s,
                                                 char *sm,
                                                 bool warm STAGE_STAMP_PARAM) {
  static_assert(
      NC % 128 == 0 && NC <= 512 && (NC == 0 || CAP > 0),
      "sparse_attn: compressed top-k in whole pairs of 64-key blocks, "
      "at most 512 (the ids' smem)");
  static_assert(D == 512 && W == 128 && RD == 64 && H % SA_HPT == 0,
                "sparse_attn: heads of 512 columns (8 warps of 64 in P V, the "
                "RoPE columns warp 7's), a window of 128 (8 warps of 16 in S, "
                "2 blocks of 64)");
  constexpr int ROW = D + D / 32; // a cache row's bytes
  constexpr int N = H * D;        // o's width
  // a Reuse layer with a task per pair (sa_bf16k above): its compressed K
  // dequantized once per token, into kbf
  constexpr bool BF16K = sa_bf16k<NC, CNODE, CCNODE, H>();
  constexpr int NP = NC / 128;          // compressed pairs
  constexpr int HALF = 64 * SA_QLD * 2; // a pair's 64 rows of K: bytes
  // split: task (t, g, s) pair s of the list; the combine's share, s's 128-
  // column blocks [b0, b1) of the group's 64
  constexpr bool SPLIT = NS > 1;
  constexpr int G = H / SA_HPT, NB = SA_HPT * D / 128;
  static_assert(!SPLIT || (NC > 0 && NS == 1 + NP && !BF16K),
                "sparse_attn: split, a task per pair of a compressed layer");
  static_assert(NS * SA_HPT * 2 * 4 <= 2 * 512 * 2,
                "sparse_attn: the combine's weights in the dequant scratch");
  float const NEG_INF = __uint_as_float(0xff800000u);
  int const tid = threadIdx.x, w = tid >> 5, l = tid & 31, qd = l & 3,
            gr = l >> 2;
  __nv_bfloat16 *const qs = reinterpret_cast<__nv_bfloat16 *>(sm + SA_OFF_Q);
  __nv_bfloat16 *const ks = reinterpret_cast<__nv_bfloat16 *>(sm + SA_OFF_K);
  __nv_bfloat16 *const ps = reinterpret_cast<__nv_bfloat16 *>(sm + SA_OFF_P);
  float *const ss = reinterpret_cast<float *>(sm + SA_OFF_S);
  float *const rs = reinterpret_cast<float *>(sm + SA_OFF_R);
  unsigned char *const es = reinterpret_cast<unsigned char *>(sm + SA_OFF_E);
  float *const os = reinterpret_cast<float *>(sm + SA_OFF_O);
  float2 *const fs = reinterpret_cast<float2 *>(sm + SA_OFF_F);
  int *const ids = reinterpret_cast<int *>(sm + SA_OFF_I);
  int const p = pos[t];
  int const oldest = p % W + 1; // list position i: slot (oldest + i) % W
  int const lo = p >= W - 1 ? 0 : W - 1 - p; // positions < lo: slots > p
  unsigned char const *const kc = cache + (size_t)t * W * ROW;
  bool const win = !SPLIT || s == 0; // this task has the window pair
  // split: the group's block maxima and sums [NS][16][4] (max of block 0, of
  // block 1, their sums; mlb [T][H / 16][NS][16][4], 0xFF before each launch)
  float *const pml =
      SPLIT ? mlb + (size_t)(t * G + g) * NS * SA_HPT * 4 : nullptr;
  // the token's window (W rows, 66 KB) into L2 first: the 4 rounds below then
  // wait for L2, not HBM (L2 is emptied before each launch; the loop's
  // rounds wait one after another)
  for (int i = tid; win && i < W * ROW / 128; i += 256) {
    asm volatile("prefetch.global.L2 [%0];" ::"l"(kc + 128 * i));
  }
  // what waits for nothing: the RoPE pairs at p (thread i < RD / 2: pair i,
  // into fs after the barrier), the head's sink (softmax thread: head tid /
  // 16)
  float2 fq;
  if (tid < RD / 2) {
    fq = reinterpret_cast<float2 const *>(freqs)[(size_t)p * (RD / 2) + tid];
  }
  float const snk = sink[g * SA_HPT + (tid >> 4)];
  __syncthreads(); // the ring area: the previous task on this SM is done with
                   // it
  if (tid < RD / 2) {
    fs[tid] = fq;
  }
  // the compressed list and its rows into L2 (each 3 lines of 128 B): graph
  // inputs (a Reuse layer) here, ready now; the list an indexer_topk's output
  // (CNODE) or the cache ckv_post's (CCNODE): after the inputs' wait of the
  // real pass. A split task: its pair's 128 (list positions i0 .. i0 + 127)
  int const i0 = SPLIT ? 128 * (s - 1) : 0, i1 = SPLIT ? 128 * s : NC;
  auto load_ids = [&](bool relaxed) {
    for (int i = i0 + tid; i < i1; i += 256) {
      int const *a = cidx + (size_t)t * NC + i;
      int id;
      if (relaxed) {
        asm volatile("ld.relaxed.gpu.global.s32 %0, [%1];"
                     : "=r"(id)
                     : "l"(a)
                     : "memory");
      } else {
        id = *a;
      }
      ids[i] = id;
      if (id >= 0) {
        unsigned char const *row = ccache + ((size_t)t * CAP + id) * SA_CROW;
#pragma unroll
        for (int k = 0; k < 3; k++) {
          asm volatile("prefetch.global.L2 [%0];" ::"l"(row + 128 * k));
        }
      }
    }
  };
  if constexpr (NC > 0 && !CNODE && !CCNODE) {
    if (!SPLIT || s > 0) {
      load_ids(false);
    }
  }
  if constexpr (BF16K) { // sa_bf16k below: the first pair's rows on their way
                         // during the window's rounds
    if (tid == 0) {
      mbar_init(su32(sm + SA_OFF_B), 1);
      mbar_init(su32(sm + SA_OFF_B) + 8, 1);
    }
    __syncthreads(); // ids
    if (g < NP) {
      sa_stage_rows<CAP>(ccache, ids, t, g, true, sm);
    }
  }
  // split, a Reuse layer's compressed pair: its K now, while q is on its way
  // (a Full layer's: once its list and rows are written, in the real pass)
  if constexpr (SPLIT) {
    if constexpr (!CNODE && !CCNODE) {
      if (s > 0) {
        __syncthreads(); // ids
        sa_stage_rows<CAP>(ccache, ids, t, s - 1, true, sm);
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncthreads();
        sa_dequant_rows(true, sm);
      }
    }
  }
// K at list positions 0 .. W - 2 (earlier launches' rows): warp w positions
// w, w + 8, ..; lane l columns 16 l .. 16 l + 15 (scale byte l / 2); 0 for
// the unwritten slots. 4 rounds of 4 rows (a round's loads in flight
// together) on one copy of the code (unroll 1): this code runs once, cold,
// and its instruction fetch is most of its time
#pragma unroll 1
  for (int h = 0; win && h < 4; h++) {
    uint4 v[4];
    uint32_t s[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int const li = w + 8 * (4 * h + i);
      if (li < W - 1 && li >= lo) {
        unsigned char const *row = kc + (size_t)((oldest + li) % W) * ROW;
        v[i] = *reinterpret_cast<uint4 const *>(row + 16 * l);
        s[i] = row[D + l / 2];
      }
    }
#pragma unroll
    for (int i = 0; i < 4; i++) {
      int const li = w + 8 * (4 * h + i);
      __nv_bfloat16 *dst = ks + li * SA_QLD + 16 * l;
      if (li < W - 1 && li >= lo) {
        sa_dequant16(v[i], s[i], dst);
      } else if (li < W - 1) {
        reinterpret_cast<uint4 *>(dst)[0] = make_uint4(0, 0, 0, 0);
        reinterpret_cast<uint4 *>(dst)[1] = make_uint4(0, 0, 0, 0);
      }
    }
  }
  // sa_bf16k: the compressed rows dequantized once per token, while q is on
  // its way: task (t, g) the pairs c = g, g + H / 16,
  // .. of token t's list (H 64: one each) -> kbf [T][NC][SA_QLD] bf16 (rows
  // laid out as in ks: a pair's half one bulk copy), then the pair's flag (the
  // node's counter 1 + NP t + c) += 1. The node's other tasks of token t run at
  // the same time (concurrent_along_axis 1): the pair loop's copies wait for
  // the flags
  if constexpr (BF16K) {
#pragma unroll 1
    for (int c = g; c < NP; c += H / SA_HPT) {
      if (c != g) { // (pair g's: staged before the window's rounds)
        sa_stage_rows<CAP>(ccache, ids, t, c, true, sm);
      }
      asm volatile("cp.async.wait_all;" ::: "memory");
      __syncthreads();
      sa_dequant_rows(true, sm, kbf + ((size_t)t * NC + 128 * c) * SA_QLD);
      // the stores before the readers' bulk copies (the async proxy)
      asm volatile("fence.proxy.async.global;" ::: "memory");
      __syncthreads(); // every thread's stores; the staging area free again
      if (tid == 0) {
        cnt_add(done_cnt + 1 + NP * t + c, 1);
      }
    }
  }
  // the dry pass's stand-ins: q and the new K row 0 (S = 0 or -inf, P = 1 or
  // 0: every value finite, so the dry pass takes the real pass's code paths,
  // not expf's or the division's slow paths)
  if (!warm) {
#pragma unroll
    for (int k = 0; k < 4; k++) {
      int const i = tid + 256 * k, r = i >> 6, c = (i & 63) * 8;
      *reinterpret_cast<uint4 *>(qs + r * SA_QLD + c) = make_uint4(0, 0, 0, 0);
    }
    if (win && tid < D / 8) {
      *reinterpret_cast<uint4 *>(ks + (W - 1) * SA_QLD + 8 * tid) =
          make_uint4(0, 0, 0, 0);
    }
    if constexpr (SPLIT && (CNODE || CCNODE)) { // K not here yet: zeros
      for (int i = tid; !win && i < 128 * (D / 8); i += 256) {
        *reinterpret_cast<uint4 *>(ks + (i >> 6) * SA_QLD + (i & 63) * 8) =
            make_uint4(0, 0, 0, 0);
      }
    }
  }
#ifdef STATIC_TIMING_BUILD
  __syncthreads();
  SA_MARK(0)
#endif
  // pass 0 (only when the code is cold: !warm, this node's first task on the
  // SM) is the dry pass: S, softmax, P V and the tail on the stand-ins above,
  // no global loads or stores. One
  // copy of the code for both passes (unroll 1): pass 0 loads the
  // instructions pass 1 runs while the inputs are on their way
#pragma unroll 1
  for (int pass = warm ? 1 : 0; pass < 2; pass++) {
    bool const real = pass == 1;
    // q from wq_b's parts, before the kv wait (a Full layer's compressed
    // pair: before its list's wait too; q usually comes before the top-k)
    sa_q_parts<H, D, RD, QP, !SPLIT>(q, freqs, qs, t, g, p, real);
    if (real) {
      SA_MARK(10)
    }
    // split, a Full layer's compressed pair: its K, once its list and rows
    // are written
    if constexpr (SPLIT && (CNODE || CCNODE)) {
      if (!win) {
        if (real) {
          if (tid == 0) {
            if constexpr (CNODE) {
              cnt_wait(ci_cnt + t, 1); // indexer_topk wrote token t's list
            }
            if constexpr (CCNODE) {
              cnt_wait(cc_cnt + t, 1); // ckv_post wrote token t's new row
            }
          }
          __syncthreads();
          SA_MARK(19)
          load_ids(CNODE);
        }
        __syncthreads(); // ids
        sa_stage_rows<CAP>(ccache, ids, t, s - 1, real, sm);
        asm volatile("cp.async.wait_all;" ::: "memory");
        __syncthreads();
        if (real) {
          SA_MARK(20)
        }
        sa_dequant_rows(real, sm);
      }
    }
    if (real) {
      if (tid == 0) {
        if (win) {
          cnt_wait(kv_cnt + t, 1); // kv_post wrote row p % W
        }
        if constexpr (NC > 0 && CNODE && !SPLIT) {
          cnt_wait(ci_cnt + t, 1); // indexer_topk wrote token t's list
        }
        if constexpr (NC > 0 && CCNODE && !SPLIT) {
          cnt_wait(cc_cnt + t, 1); // ckv_post wrote token t's new row
        }
      }
      __syncthreads();
      SA_MARK(1)
      if constexpr (NC > 0 && (CNODE || CCNODE) && !SPLIT) { // before the
                                                             // barrier ahead
                                                             // of pair 0
        load_ids(CNODE);
      }
      // K at list position W - 1: slot p % W, this launch's row (warp 0;
      // relaxed loads: the row shares 128-B lines with rows read above, maybe
      // in L1)
      if (win && w == 0) {
        unsigned char const *row = kc + (size_t)(p % W) * ROW;
        uint4 const v = ld16_relaxed(row + 16 * l);
        uint4 const sv = ld16_relaxed(row + D);
        int const b = l / 2; // the scale byte
        uint32_t const sw = b < 4 ? sv.x : b < 8 ? sv.y : b < 12 ? sv.z : sv.w;
        sa_dequant16(
            v, (sw >> (8 * (b & 3))) & 0xFFu, ks + (W - 1) * SA_QLD + 16 * l);
      }
    }
    __syncthreads();
    if (real) {
      SA_MARK(2)
    }
    // the list in pairs of 64-position blocks (the official's blocks, in
    // order): pair 0 the window (K in ks since the prologue), pair c > 0 the
    // compressed rows of list positions 128 (c - 1) .. + 127 (ids), K
    // dequantized into ks after the previous pair's P V (sa_bf16k: copied from
    // kbf, half 0 during block 1's P V, half 1 after it). Per head the running
    // max mr (from -1e30) and sum sr carry from pair to pair, O in acc
    float acc[8][4] = {};
    float mr = -1.0e30f, sr = 0.f; // the softmax thread's head's
    float bs0 = 0.f, bs1 = 0.f;    // (split: the pair's block sums)
    // (split: pair s alone, its K in ks since the prologue or the pass start)
    int const npair = SPLIT ? s + 1 : 1 + (real ? NC / 128 : (NC > 0 ? 1 : 0));
#pragma unroll 1
    for (int pr = SPLIT ? s : 0; pr < npair; pr++) {
      if constexpr (!BF16K) {
        if (pr + 1 < npair) { // the next pair's rows: raw bytes into the
                              // staging area (cp.async: no registers held)
          sa_stage_rows<(CAP > 0 ? CAP : 1)>(ccache, ids, t, pr, real, sm);
        }
      }
      // S = q K^T x scale, warp w pair positions 16 w .. 16 w + 15 (2 n8
      // tiles); -inf where the list holds nothing (an unwritten window slot,
      // a compressed id -1)
      {
        // keys per warp: 32 (warps 0..3, q's fragments read 4 times, not 8)
        // without an indexer list or a compressed row written this launch;
        // 16 (all 8 warps) with them (that layer's kernel spills with 32)
        constexpr int KW = (CNODE || CCNODE) ? 16 : 32;
        constexpr int NT = KW / 8; // the warp's n8 tiles
        bool const sw = KW * w < 128;
        float acc_s[NT][4] = {};
        if constexpr (BF16K) { // warp w's rows: half KW w / 64 of the copy
          if (real && pr > 0 && sw) {
            mbar_wait(su32(sm + SA_OFF_B) + 8 * (KW * w / 64), (pr - 1) & 1);
          }
        }
        if (sw && (pr > 0 || KW * w + KW - 1 >= lo)) {
          uint32_t const qa = su32(qs) + ((l & 15) * SA_QLD + (l >> 4) * 8) * 2;
          uint32_t const kb =
              su32(ks) + ((KW * w + (l >> 4) * 8 + (l & 7)) * SA_QLD +
                          ((l >> 3) & 1) * 8) *
                             2;
#pragma unroll 8
          for (int k = 0; k < D / 16; k++) {
            uint32_t a[4];
            sa_ldsm_x4(qa + k * 32, a);
#pragma unroll
            for (int tp = 0; tp < NT / 2; tp++) {
              uint32_t b[4];
              sa_ldsm_x4(kb + (tp * 16 * SA_QLD + k * 16) * 2, b);
              sa_mma(acc_s[2 * tp], a, b[0], b[1]);
              sa_mma(acc_s[2 * tp + 1], a, b[2], b[3]);
            }
          }
        }
        float const scale = __int_as_float(SCALE_BITS);
#pragma unroll
        for (int j = 0; j < NT; j++) {
          if (!sw) {
            break;
          }
          int const li = KW * w + 8 * j + 2 * qd;
          bool v0 = li >= lo, v1 = li + 1 >= lo;
          if (pr > 0) {
            v0 = ids[128 * (pr - 1) + li] >= 0;
            v1 = ids[128 * (pr - 1) + li + 1] >= 0;
          }
#pragma unroll
          for (int hh = 0; hh < 2; hh++) {
            float2 v;
            v.x = v0 ? __fmul_rn(acc_s[j][2 * hh], scale) : NEG_INF;
            v.y = v1 ? __fmul_rn(acc_s[j][2 * hh + 1], scale) : NEG_INF;
            *reinterpret_cast<float2 *>(ss + (gr + 8 * hh) * SA_SLD + li) = v;
          }
        }
      }
      __syncthreads();
      if (real && pr == 0) {
        SA_MARK(3)
      }
      if (real && pr == 1) {
        SA_MARK(11)
      }
      // softmax: thread (head r, block b, key half wh, lane quarter qd) takes
      // the block's positions 32 wh + 2 qd + 8 k + e (k < 4, e < 2), as the
      // official's lane: its 8 values in order e = 0 then 1, k = 0..3. m1, m2:
      // the running max after the pair's blocks 0, 1; the rescales
      // sc0 = exp(mr - m1) (O and the sum before block 0), sc1 = exp(m1 - m2)
      {
        int const r = tid >> 4, b = (tid >> 3) & 1, wh = (tid >> 2) & 1;
        int const base = 64 * b + 32 * wh + 2 * qd;
        float x[8];
#pragma unroll
        for (int k = 0; k < 4; k++) {
          float2 const v =
              *reinterpret_cast<float2 const *>(ss + r * SA_SLD + base + 8 * k);
          x[k] = v.x;
          x[4 + k] = v.y;
        }
        float mx = x[0];
#pragma unroll
        for (int i = 1; i < 8; i++) {
          mx = fmaxf(mx, x[i]);
        }
        for (int d = 1; d < 8; d <<= 1) { // the block's 8 threads of head r
          mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, d));
        }
        float const mo = __shfl_xor_sync(0xffffffffu, mx, 8); // the other block
        if constexpr (SPLIT) { // the running max before this pair: the earlier
                               // pairs' block maxima (tasks (t, g, 0 .. s - 1))
          if (real) {
            if ((tid & 15) == 0) { // (b = 0: mx block 0's, mo block 1's)
              *reinterpret_cast<float2 *>(pml + (s * SA_HPT + r) * 4) =
                  make_float2(mx, mo);
            }
            // the earlier pairs' maxima of head r, polled (all loads of a
            // round in flight) until written (a max is never 0xFFFFFFFF)
            float2 v[NS - 1];
            bool ok[NS - 1];
#pragma unroll
            for (int c = 0; c < NS - 1; c++) {
              ok[c] = c >= s;
            }
            bool all;
            do {
              all = true;
#pragma unroll
              for (int c = 0; c < NS - 1; c++) {
                if (!ok[c]) {
                  asm volatile("ld.relaxed.gpu.global.v2.f32 {%0, %1}, [%2];"
                               : "=f"(v[c].x), "=f"(v[c].y)
                               : "l"(pml + (c * SA_HPT + r) * 4));
                }
              }
#pragma unroll
              for (int c = 0; c < NS - 1; c++) {
                if (!ok[c]) {
                  ok[c] = validf(v[c].x) && validf(v[c].y);
                }
                all = all && ok[c];
              }
            } while (!all);
#pragma unroll
            for (int c = 0; c < NS - 1; c++) {
              if (c < s) {
                mr = fmaxf(mr, fmaxf(v[c].x, v[c].y));
              }
            }
          }
        }
        float const m1 = fmaxf(mr, b ? mo : mx), m2 = fmaxf(m1, b ? mx : mo);
        float const mb = b ? m2 : m1;
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
          x[i] = expf(__fsub_rn(x[i], mb));
          s = __fadd_rn(s, x[i]);
        }
#pragma unroll
        for (int k = 0; k < 4; k++) {
          *reinterpret_cast<__nv_bfloat162 *>(ps + r * SA_PLD + base + 8 * k) =
              __floats2bfloat162_rn(x[k], x[4 + k]);
        }
        s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 4)); // 2 key halves
        s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 2)); // lane quarters
        s = __fadd_rn(s, __shfl_xor_sync(0xffffffffu, s, 1));
        float const so = __shfl_xor_sync(0xffffffffu, s, 8);
        float const sum0 = b ? so : s, sum1 = b ? s : so;
        bs0 = sum0;
        bs1 = sum1;
        float const sc0 = expf(__fsub_rn(mr, m1)),
                    sc1 = expf(__fsub_rn(m1, m2));
        sr = __fmaf_rn(__fmaf_rn(sr, sc0, sum0), sc1, sum1);
        mr = m2;
        if ((tid & 15) == 0) {
          rs[4 * r] = sc0;
          rs[4 * r + 1] = sc1;
          rs[4 * r + 2] = __fadd_rn(sr, expf(__fsub_rn(snk, mr))); // the last
                                                                   // pair's
        }
      }
      __syncthreads();
      if (real && pr == 0) {
        SA_MARK(4)
      }
      if (real && pr == 1) {
        SA_MARK(12)
      }
      // O = O x sc0 + P_0 V_0, O = O x sc1 + P_1 V_1: warp w columns 64 w ..
      // 64 w + 63 (8 n8 tiles), each block's 4 k-steps in order; in the window
      // pair a k-step of only unwritten positions (P and V 0) adds nothing
      {
        uint32_t const pa = su32(ps) + ((l & 15) * SA_PLD + (l >> 4) * 8) * 2;
        uint32_t const vb =
            su32(ks) +
            ((((l >> 3) & 1) * 8 + (l & 7)) * SA_QLD + 64 * w + (l >> 4) * 8) *
                2;
#pragma unroll
        for (int blk = 0; blk < 2; blk++) {
          if constexpr (BF16K) { // rows 0 .. 63 read: the next pair's half 0
            if (blk == 1 && real && pr + 1 < npair) {
              asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
              __syncthreads();
              if (tid == 0) {
                sa_copy_half<NC, NP, HALF>(kbf, done_cnt, t, pr, 0, sm);
              }
            }
          }
          float const a1 = rs[4 * gr + blk], a2 = rs[4 * (gr + 8) + blk];
#pragma unroll
          for (int j = 0; j < 8; j++) {
            acc[j][0] = __fmul_rn(acc[j][0], a1);
            acc[j][1] = __fmul_rn(acc[j][1], a1);
            acc[j][2] = __fmul_rn(acc[j][2], a2);
            acc[j][3] = __fmul_rn(acc[j][3], a2);
          }
#pragma unroll
          for (int kk = 0; kk < 4; kk++) {
            int const k = 4 * blk + kk;
            if (pr == 0 && 16 * k + 15 < lo) {
              continue;
            }
            uint32_t a[4];
            sa_ldsm_x4(pa + k * 32, a);
#pragma unroll
            for (int jj = 0; jj < 4; jj++) {
              uint32_t b[4];
              sa_ldsm_x4_t(vb + (k * 16 * SA_QLD + 16 * jj) * 2, b);
              sa_mma(acc[2 * jj], a, b[0], b[1]);
              sa_mma(acc[2 * jj + 1], a, b[2], b[3]);
            }
          }
        }
      }
      if (real && pr == 1) {
        SA_MARK(13)
      }
      if (pr + 1 < npair) {    // the next pair's K: after every warp's P V
        if constexpr (BF16K) { // half 1 (the dry pass: ks keeps the window's)
          asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
          __syncthreads();
          if (real && pr == 1) {
            SA_MARK(15)
          }
          if (real && tid == 0) {
            sa_copy_half<NC, NP, HALF>(kbf, done_cnt, t, pr, 1, sm);
          }
        } else {
          asm volatile("cp.async.wait_all;" ::: "memory");
          __syncthreads();
          if (real && pr == 1) {
            SA_MARK(15)
          }
          sa_dequant_rows(real, sm);
          __syncthreads();
        }
      }
      if (real && pr == 1) {
        SA_MARK(14)
      }
    }
    if (real) {
      SA_MARK(8)
    }
    // O -> fp32 rows os [16][SA_OSLD]
#pragma unroll
    for (int j = 0; j < 8; j++) {
      int const c = 64 * w + 8 * j + 2 * qd;
      *reinterpret_cast<float2 *>(os + gr * SA_OSLD + c) =
          make_float2(acc[j][0], acc[j][1]);
      *reinterpret_cast<float2 *>(os + (gr + 8) * SA_OSLD + c) =
          make_float2(acc[j][2], acc[j][3]);
    }
    __syncthreads();
    // the tail: the 256 32-column groups (16 heads x 16) over the 64 thread
    // quads, 4 rounds; quad thread q4 columns 8 q4 .. 8 q4 + 7 of the group (a
    // half-warp 128 contiguous floats: no bank conflicts). o = O / denominator
    // -> bf16; the last RD columns: the inverse RoPE (torch's complex multiply
    // by conj(freqs_cis)) -> bf16; act_quant's formula per group (its amax over
    // the quad) -> o_q's bytes, the scale byte (st: stored; every lane runs it,
    // the quad's shuffles). A loop, not unrolled copies: this code runs cold
    // once per SM (its instruction fetch is its cost)
    int const q4 = tid & 3;
    auto quant8 = [&](int r, int gi, float(&x)[8], float den, bool st) {
      int const c = 32 * gi + 8 * q4;
#pragma unroll
      for (int e = 0; e < 8; e++) {
        x[e] = bf16r(__fdiv_rn(x[e], den));
      }
      if (c >= D - RD) {
#pragma unroll
        for (int e = 0; e < 8; e += 2) {
          float2 const f = fs[(c + e - (D - RD)) / 2]; // (cos, sin)
          float const cr = f.x, ci = -f.y;             // conj(freqs_cis)
          float const a = x[e], b = x[e + 1];
          x[e] = bf16r(__fmaf_rn(a, cr, -__fmul_rn(b, ci)));
          x[e + 1] = bf16r(__fmaf_rn(b, cr, __fmul_rn(a, ci)));
        }
      }
      float amax = 0.f;
#pragma unroll
      for (int e = 0; e < 8; e++) {
        amax = fmaxf(amax, fabsf(x[e]));
      }
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
      int const ex = xq_exponent(amax);
      uint2 const qb = make_uint2(xq_pack4(x, ex), xq_pack4(x + 4, ex));
      if (real && st) {
        *reinterpret_cast<uint2 *>(out + (size_t)t * N +
                                   (size_t)(g * SA_HPT + r) * D + c) = qb;
      }
      if (q4 == 0 && st) {
        es[r * 16 + gi] = (unsigned char)(ex + 127);
      }
    };
    int lb0 = 0, lnb = NB; // the 128-column blocks whose scales this task
                           // writes
    if constexpr (!SPLIT) {
#pragma unroll 1
      for (int it = 0; it < 4; it++) {
        int const gq = (tid >> 2) + 64 * it, r = gq >> 4, gi = gq & 15;
        int const c = 32 * gi + 8 * q4;
        float4 const v0 = *reinterpret_cast<float4 const *>(os + r * SA_OSLD +
                                                            c),
                     v1 = *reinterpret_cast<float4 const *>(os + r * SA_OSLD +
                                                            c + 4);
        float x[8] = {v0.x, v0.y, v0.z, v0.w, v1.x, v1.y, v1.z, v1.w};
        quant8(r, gi, x, rs[4 * r + 2], true);
      }
    } else {
      // split: this task's partial (O from os; the block sums of each head:
      // its softmax threads') -> the group's partials po [NS][16][D] but its
      // own share's blocks (they stay in os), then (after the barrier: issued
      // after the partial's) the sums into pml [NS][16][4] (2, 3): plain
      // stores into 0xFF, the readers poll them (data-as-flag: no count, no
      // bulk copies). Then the group's sums (16 threads) and the NS
      // partials' shares of blocks [b0, b1) polled in one loop, into
      // registers (each value checked, re-loaded if not there yet; the
      // share's one reader: REARM, it re-arms them after the node's count,
      // below). The tail on
      // the combined share (the dry pass: O's values from os, maxima 0, sums
      // 1)
      int const b0 = s * NB / NS, b1 = (s + 1) * NB / NS, nb = b1 - b0;
      size_t const grp = (size_t)(t * G + g) * NS;
      float *const po = part + grp * (SA_HPT * D);
      float *const fac = reinterpret_cast<float *>(sm + SA_OFF_Y); // [NS][16]
      if (real) {
        for (int i = tid; i < SA_HPT * D / 4; i += 256) {
          int const r = i / (D / 4), c = 4 * (i % (D / 4)),
                    b = (r * D + c) / 128;
          if (b < b0 || b >= b1) {
            *reinterpret_cast<float4 *>(po + ((size_t)s * SA_HPT + r) * D + c) =
                *reinterpret_cast<float4 const *>(os + r * SA_OSLD + c);
          }
        }
      }
      __syncthreads();
      if (real) {
        if ((tid & 15) == 0) {
          *reinterpret_cast<float2 *>(pml + (s * SA_HPT + (tid >> 4)) * 4 + 2) =
              make_float2(bs0, bs1);
        }
      }
      if (real) {
        SA_MARK(16)
      }
      // one polling loop for the group's sums (thread 16 rh: head rh's) and
      // the share's partials (every quad: the share's 32-column groups 4 b0 ..
      // 4 b1 - 1; the quads past them the share's first group, nothing polled
      // or stored), their loads in flight together; then per head the
      // official's chain over the pairs (-> fac, the denominator)
      int const qq = tid >> 2, act = qq < 4 * nb, gq = 4 * b0 + (act ? qq : 0),
                r = gq >> 4, gi = gq & 15, rh = tid >> 4;
      bool const sumt = (tid & 15) == 0;
      float4 v[NS][2], sv[NS];
      {
        float const *const src = po + 32 * gq + 8 * q4; // + s2 16 D
        float const *const own = os + r * SA_OSLD + 32 * gi + 8 * q4;
        bool ok[NS], oks[NS];
#pragma unroll
        for (int s2 = 0; s2 < NS; s2++) {
          ok[s2] = !real || !act || s2 == s;
          oks[s2] = !real || !sumt;
          v[s2][0] = *reinterpret_cast<float4 const *>(own);
          v[s2][1] = *reinterpret_cast<float4 const *>(own + 4);
          sv[s2] = make_float4(0.f, 0.f, 1.f, 1.f);
        }
        bool all;
        do {
#pragma unroll
          for (int s2 = 0; s2 < NS; s2++) {
            if (!oks[s2]) {
              sv[s2] = ldf4_poll(pml + (s2 * SA_HPT + rh) * 4);
            }
            if (!ok[s2]) {
              v[s2][0] = ldf4_poll(src + (size_t)s2 * SA_HPT * D);
              v[s2][1] = ldf4_poll(src + (size_t)s2 * SA_HPT * D + 4);
            }
          }
          all = true;
#pragma unroll
          for (int s2 = 0; s2 < NS; s2++) {
            if (!oks[s2]) {
              oks[s2] = validf4(sv[s2]);
            }
            if (!ok[s2]) {
              ok[s2] = validf4(v[s2][0]) && validf4(v[s2][1]);
            }
            all = all && ok[s2] && oks[s2];
          }
        } while (!all);
      }
      if (sumt) {
        float m = -1.0e30f, sum = 0.f;
#pragma unroll
        for (int s2 = 0; s2 < NS; s2++) {
          float const m1 = fmaxf(m, sv[s2].x), m2 = fmaxf(m1, sv[s2].y);
          float const sc0 = expf(__fsub_rn(m, m1)),
                      sc1 = expf(__fsub_rn(m1, m2));
          sum = __fmaf_rn(__fmaf_rn(sum, sc0, sv[s2].z), sc1, sv[s2].w);
          *reinterpret_cast<float2 *>(fac + (s2 * SA_HPT + rh) * 2) =
              make_float2(sc0, sc1);
          m = m2;
        }
        rs[4 * rh + 3] = __fadd_rn(sum, expf(__fsub_rn(snk, m)));
      }
      __syncthreads(); // fac, the denominators
      if (real) {
        SA_MARK(18)
      }
      // O = pair 0's, then O = O sc0 sc1 + pair s2's (pair 0's: the
      // official's bits)
      {
        float x[8];
#pragma unroll
        for (int s2 = 0; s2 < NS; s2++) {
          float2 const f =
              *reinterpret_cast<float2 const *>(fac + (s2 * SA_HPT + r) * 2);
          float const y[8] = {v[s2][0].x,
                              v[s2][0].y,
                              v[s2][0].z,
                              v[s2][0].w,
                              v[s2][1].x,
                              v[s2][1].y,
                              v[s2][1].z,
                              v[s2][1].w};
#pragma unroll
          for (int e = 0; e < 8; e++) {
            x[e] = s2 == 0
                       ? y[e]
                       : __fadd_rn(__fmul_rn(__fmul_rn(x[e], f.x), f.y), y[e]);
          }
        }
        quant8(r, gi, x, rs[4 * r + 3], act);
      }
      lb0 = b0;
      lnb = nb;
    }
    if (real) {
      SA_MARK(9)
    }
    __syncthreads();
    if (real) {
      SA_MARK(5)
      // the scale chunks of heads [16 g, 16 g + 16) (64; a split task its
      // share's): a 16-B line each
      if ((tid >> 2) < lnb) {
        int const lc = lb0 + (tid >> 2); // head lc / 4, its 128-column block
                                         // lc % 4
        uint32_t const s4 = *reinterpret_cast<uint32_t const *>(
            es + (lc >> 2) * 16 + (lc & 3) * 4);
        xq_scale_line(out, N, g * (SA_HPT * D / 128) + lc, t, tid & 3, s4);
      }
    } else {
      SA_MARK(7)
    }
  }
  if constexpr (BF16K) { // every copy waited for (the last pair's S)
    if (tid == 0) {
      asm volatile(
          "mbarrier.inval.shared::cta.b64 [%0];" ::"r"(su32(sm + SA_OFF_B))
          : "memory");
      asm volatile(
          "mbarrier.inval.shared::cta.b64 [%0];" ::"r"(su32(sm + SA_OFF_B) + 8)
          : "memory");
    }
  }
  // the ring area's next TMA writes come after these generic stores
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  SA_MARK(6)
  if (tid == 0) {
    cnt_add(done_cnt, 1); // release: the CTA's stores before (the barrier)
  }
  if constexpr (SPLIT && REARM) {
    // split: the share's re-arms after the count (its release waits for the
    // CTA's stores before it, not these; the next writers are the next
    // launch's): the other partials' share values this task polled, and q's
    // parts of its share (every split read them before its maxima)
    int const b0 = s * NB / NS, nb = (s + 1) * NB / NS - b0, qq = tid >> 2;
    if (qq < 4 * nb) {
      float *const src = part + (size_t)(t * G + g) * NS * (SA_HPT * D) +
                         32 * (4 * b0 + qq) + 8 * (tid & 3);
#pragma unroll
      for (int s2 = 0; s2 < NS; s2++) {
        if (s2 != s) {
          rearm16(src + (size_t)s2 * SA_HPT * D);
          rearm16(src + (size_t)s2 * SA_HPT * D + 4);
        }
      }
    }
    float const *src = q + (size_t)t * N + (size_t)g * SA_HPT * D + 128 * b0;
    for (int i = tid; i < QP * nb * 32; i += 256) {
      int const pp = i / (nb * 32), v4 = i - pp * (nb * 32);
      rearm16(src + (size_t)pp * T * N + 4 * v4);
    }
  }
}

// sparse_attn task (t, g) of grid (T, H / 16); Q: wq_b's node (a gemm_fp8_tile
// in slots, 1 or 2 K parts: sa_q_parts), KV: kv_post's
// (the window cache, a graph tensor; params {D, W, RD, eps}; a counter per
// token). PARAMS {H, D, W, RD, softmax scale (fp32 bits)}; BUF_SLOTS {the
// window cache
// ([T][W][D + D / 32] uint8), positions ([T] int32), freqs ([rows][RD /
// 2][2] fp32), attn_sink ([H] fp32)}, graph tensors; warm: this node ran on
// this SM before (no dry pass)
// the counter of compress_idx's producer (the 5th input after the window
// cache: positions, freqs, attn_sink, the compressed cache, compress_idx): an
// indexer_topk's; -1 for a graph input or without compressed KV. The same for
// the compressed cache's (the 4th: ckv_post's). Both graph inputs (a Reuse
// layer): BUF_SLOTS' 7th the bf16 K buffer [T][NC][SA_QLD] (a scratch buffer),
// and the node's counters 1 + NP t + c the pairs' flags (sa_bf16k). Grid (T, H
// / 16, NS), NS = 1 + NC / 128 (split): BUF_SLOTS' 7th the partials' O [T][H /
// 16][NS][16][D] fp32, 8th their block maxima and sums [T][H / 16][NS][16][4]
// fp32 (0xFF before each launch: the maxima polled), scratch buffers; the
// node's counters 1 + t H / 16 + g the groups' counts
template <int I, class X0, class... XS>
struct sa_nth {
  using type = typename sa_nth<I - 1, XS...>::type;
};
template <class X0, class... XS>
struct sa_nth<0, X0, XS...> {
  using type = X0;
};
template <int I, class... REST>
__host__ __device__ constexpr int sa_rest_counter() {
  if constexpr (sizeof...(REST) >= 5) {
    return sa_nth<I, REST...>::type::counter;
  } else {
    return -1;
  }
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class Q,
          class KV,
          class... REST>
__device__ __forceinline__ void run_sparse_attn(Maps const &,
                                                G const &g,
                                                KernelLocals &L,
                                                StaticTask const &tk,
                                                bool warm) {
  constexpr int H = PARAMS::v[0], D = PARAMS::v[1], W = PARAMS::v[2],
                RD = PARAMS::v[3], NC = param_or0<PARAMS, 5>(),
                CAP = param_or0<PARAMS, 6>();
  constexpr int NS = SELF::z;
  static_assert(SELF::x == T && SELF::y * SA_HPT == H &&
                    (NS == 1 || (NC > 0 && NS == 1 + NC / 128)),
                "sparse_attn: grid (T, H / 16) or (T, H / 16, 1 + NC / 128)");
  unsigned char const *ccache = nullptr;
  int const *cidx = nullptr;
  __nv_bfloat16 *kbf = nullptr;
  float *part = nullptr, *mlb = nullptr;
  if constexpr (NC > 0) {
    ccache = buf_at<unsigned char const>(g, slot_at<BUF_SLOTS, 4>());
    cidx = buf_at<int const>(g, slot_at<BUF_SLOTS, 5>());
    if constexpr (sa_bf16k<NC,
                           (sa_rest_counter<4, REST...>() >= 0),
                           (sa_rest_counter<3, REST...>() >= 0),
                           H>()) {
      kbf = buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 6>());
    }
    if constexpr (NS > 1) {
      part = buf_at<float>(g, slot_at<BUF_SLOTS, 6>());
      mlb = buf_at<float>(g, slot_at<BUF_SLOTS, 7>());
    }
  }
  static_assert(SELF::buf >= 0 && exchange_bytes[SELF::buf] == 0 &&
                    SELF::counter >= 0,
                "sparse_attn: its output in this GPU's memory, a counter of "
                "its tasks");
  // q: wq_b's K parts in slots (a gemm_fp8_tile's params {COMBINE_SLOTS, K,
  // H D, ..})
  static_assert(Q::v[0] == COMBINE_SLOTS && Q::v[2] == H * D && Q::buf >= 0 &&
                    Q::y >= 1 && Q::y <= 2,
                "sparse_attn: q is wq_b's K parts in slots (at most 2)");
  static_assert(KV::buf < 0 && KV::counter >= 0 && KV::v[0] == D &&
                    KV::v[1] == W && KV::x == T && KV::y == 1 && KV::z == 1,
                "sparse_attn: the window cache is kv_post's, one task (and "
                "counter) per token");
  sparse_attn_task<H,
                   D,
                   W,
                   RD,
                   Q::y,
                   PARAMS::v[4],
                   NC,
                   CAP,
                   (sa_rest_counter<4, REST...>() >= 0),
                   (sa_rest_counter<3, REST...>() >= 0),
                   NS>(buf_at<float const>(g, Q::buf),
                       buf_at<unsigned char const>(g, slot_at<BUF_SLOTS, 0>()),
                       buf_at<int const>(g, slot_at<BUF_SLOTS, 1>()),
                       buf_at<float const>(g, slot_at<BUF_SLOTS, 2>()),
                       buf_at<float const>(g, slot_at<BUF_SLOTS, 3>()),
                       ccache,
                       cidx,
                       kbf,
                       part,
                       mlb,
                       buf_at<unsigned char>(g, SELF::buf),
                       g.cnt + KV::counter,
                       g.cnt + (sa_rest_counter<4, REST...>() >= 0
                                    ? sa_rest_counter<4, REST...>()
                                    : 0),
                       g.cnt + (sa_rest_counter<3, REST...>() >= 0
                                    ? sa_rest_counter<3, REST...>()
                                    : 0),
                       g.cnt + SELF::counter,
                       tk.x,
                       tk.y,
                       tk.z,
                       L.sm,
                       warm STAGE_STAMP_ARG(L.stage_stamps));
}

// dynamic shared memory: q, K, P, S and the rest, inside the ring area
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_sparse_attn() {
  return SA_SMEM;
}

} // namespace static_mk
