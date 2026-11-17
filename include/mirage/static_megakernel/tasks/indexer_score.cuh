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

constexpr int IS_QLD = IX_HD + 8; // q's row stride in shared memory (bf16)
constexpr int IS_OFF_Q = 0,       // q [32][IS_QLD] bf16
    IS_OFF_K = IS_OFF_Q + 32 * IS_QLD * 2; // the chunk's index K rows (raw)
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
                                                   unsigned char const *cand,
                                                   int const *pos,
                                                   __nv_bfloat16 *out,
                                                   uint32_t const *q_cnt,
                                                   uint32_t const *k_cnt,
                                                   uint32_t *done_cnt,
                                                   int t,
                                                   int c,
                                                   char *sm) {
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
    // the chunk's halves below n (inside CAP): their rows, contiguous bytes
    // (cp.async.cg: L2)
    int const nh =
        CH == 256 ? 1 : min(CH, min((n + 255) / 256 * 256, CAP) - k0) / 256;
    {
      uint32_t const dst = su32(sm + IS_OFF_K);
      unsigned char const *src = kc + ((size_t)t * CAP + k0) * IX_ROW;
      for (int i = tid; i < nh * 256 * IX_ROW / 16; i += 256) {
        asm volatile(
            "cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst + 16 * i),
            "l"(src + 16 * i)
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
          unsigned char const *row = kh + (32 * w + 8 * nt + g8) * IX_ROW;
          float const s = __uint_as_float((uint32_t)row[HD / 2 + ks / 2] << 23);
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

// indexer_score task (t, c), grid (T, NCH): chunk c of is_ch(CAP, NCH); QI:
// indexer_q's node (its buffer q_idx, a counter per token); WP: weights_proj's
// node (a gemm_tile in slots: WP::y K parts of WP::v[2] rows, the first H
// used); KC: the index K cache's producer (index_k_post, a counter per token;
// or a graph input: none to wait for). PARAMS {H, HD, CAP, R, CAND}; BUF_SLOTS
// {the index K cache
// ([T][CAP][IX_ROW] uint8), positions ([T] int32), with CAND the candidate
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
    cand = buf_at<unsigned char const>(g, slot_at<BUF_SLOTS, 2>());
  }
  indexer_score_task<H, HD, CAP, R, CAND, WP::y, WP::v[2], KNODE, SELF::y>(
      buf_at<__nv_bfloat16 const>(g, QI::buf),
      buf_at<float const>(g, WP::buf),
      buf_at<unsigned char const>(g, slot_at<BUF_SLOTS, 0>()),
      cand,
      buf_at<int const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<__nv_bfloat16>(g, SELF::buf),
      g.cnt + QI::counter,
      g.cnt + (KNODE ? KC::counter : 0),
      g.cnt + SELF::counter,
      tk.x,
      tk.y,
      L.sm);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_indexer_score() {
  return is_off_w(is_ch(PARAMS::v[2], SELF::y)) + 32 * 4;
}

} // namespace static_mk
