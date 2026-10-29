// tasks/hc_post_mix.cuh -- mHC's post step of a TP sublayer and the first half
// of the next sublayer's pre step in one task (hc_post MODE 1 and the partial
// mixes hc_pre adds): task s of grid (H / 32, 1, 1) = columns [32 s, 32 s + 32)
// of every stream of every token:
//   1. before the sums land: fn's slice (the next sublayer's hc_*_fn, its
//      NMIX rows x the task's columns of every stream) into registers, then to
//      shared memory; warp w = tokens t = w, w + 8, ..: this lane's R[t][i] (HC
//      values) and token t's coefficients
//      (h_post [T][HC], h_res [T][HC HC] fp32: the sublayer's, finished,
//      SGLang's MHCState)
//   2. warp w, lanes 0 .. 3 land the GPUs' bf16 partial sums of its tokens, 8
//      columns each (an allreduce_send node's buffer, core.cuh
//      allreduce_land_bf16: fp32, GPU order) -> out (bf16), shared
//   3. warp w, lane l: R'[t][j][c] = bf16(post[j] out[c] + sum_i comb[i][j]
//      R[t][i][c]), rounded as SGLang's _mhc_post_torch (each product, then
//      the sum over i in order, then post out added: no FMA), to R' (a graph
//      tensor) and to shared memory
//   4. the partial mixes of the task's columns as a GEMM of its slice: mixes
//      [T][NMIX] = R' [T][HC 32] (bf16, exact: R' is bf16) x fn [HC 32][NMIX]
//      (fp32 as bf16 hi + lo, two MMAs: ~2^-17 relative per product),
//      mma.sync m16n8k16, a warp per 16 tokens x 8 rows; entry NMIX the sum of
//      squares from R' in shared memory (256 / T threads per token), then 0s
//      up to HC_MIX_STRIDE -> row (s, t) of the output ([KS][T][HC_MIX_STRIDE]
//      fp32, 0xFF-prefilled, which hc_pre polls). (A warp-sum
//      form of step 4 ran latency-bound at 8 warps a SM: 14.0 / 5.4 us at
//      T = 32 / 16.)
// Counter: one; every task adds 1 (hc_pre reads R' once it is the task count).
#pragma once
#include "../core.cuh"
#include "hc_pre.cuh"

namespace static_mk {

// the bf16 row of R' / fn of a task in shared memory: HC 32
// values + 8 (a row is 4 banks after the previous one: the fragment loads of
// 8 rows x 4 lanes hit 32 banks)
template <int HC>
__host__ __device__ constexpr int hcpm_kp() {
  return HC * 32 + 8;
}
__device__ __forceinline__ void
    hcpm_mma(uint32_t const (&a)[4], uint32_t const (&b)[2], float (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t hcpm_ld2(__nv_bfloat16 const *p) {
  return *reinterpret_cast<uint32_t const *>(p); // 2 bf16 (4-B aligned)
}

template <size_t SOFF, int H, int HC, int NMIX> // SOFF: the sent sums' offset
__device__ __forceinline__ void
    hc_post_mix_task(G const &g,
                     float const *h_post,
                     float const *h_res,
                     __nv_bfloat16 const *res,
                     __nv_bfloat16 *rout,
                     float const *fn,
                     float *mix,
                     uint32_t *done_cnt,
                     int s,
                     char *scratch STAGE_STAMP_PARAM) {
  constexpr size_t S_RANK = (size_t)T * H * 2; // one GPU's slot
  constexpr int C = 32, CB = HC * HC;
  constexpr int TB = T / 8; // token blocks: warp w takes tokens w + 8 b
  static_assert(T % 8 == 0 && T <= 32 && 32 % T == 0 && H % C == 0 &&
                    C == HC_MIX_STRIDE && NMIX + 1 <= HC_MIX_STRIDE,
                "hc_post_mix: a warp per 8 tokens, a lane per column");
  // scratch, from T HC C floats on (the area below is not used): the sums
  // [T][C] fp32, R' [T][KP] bf16, fn's slice as bf16 hi and lo [NMIX][KP] (row
  // k = stream j, column c: k = 32 j + c)
  float *out_s = reinterpret_cast<float *>(scratch) + T * HC * C;
  constexpr int KP = hcpm_kp<HC>(), FNV = NMIX * HC * C / 256;
  static_assert(NMIX % 8 == 0 && (NMIX * HC * C) % 256 == 0 &&
                    ((T + 15) / 16) * (NMIX / 8) <= 8 &&
                    HC_MIX_STRIDE - NMIX <= 256 / T,
                "hc_post_mix: whole 8-row tiles of the mixes, a warp per tile, "
                "the entries after the mixes by the threads of a token");
  __nv_bfloat16 *rb = reinterpret_cast<__nv_bfloat16 *>(out_s + T * C);
  __nv_bfloat16 *fh = rb + T * KP, *fl = fh + NMIX * KP;
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int const c = s * C + lane;
  // 1. what does not wait for the sums (graph tensors: in flight while the
  // sums land)
  float fv[FNV]; // fn values idx = tid + 256 q
#pragma unroll
  for (int q = 0; q < FNV; q++) {
    int const idx = threadIdx.x + 256 * q, r = idx / (HC * C),
              k = idx % (HC * C);
    fv[q] =
        __ldg(fn + (size_t)r * HC * H + (size_t)(k / C) * H + s * C + k % C);
  }
  float rv[TB][HC], post[TB][HC], comb[TB][HC][HC];
#pragma unroll
  for (int b = 0; b < TB; b++) {
    int const t = warp + 8 * b;
#pragma unroll
    for (int i = 0; i < HC; i++) {
      rv[b][i] = __bfloat162float(res[((size_t)t * HC + i) * H + c]);
      post[b][i] = h_post[(size_t)t * HC + i];
#pragma unroll
      for (int j = 0; j < HC; j++) {
        comb[b][i][j] = h_res[(size_t)t * CB + i * HC + j];
      }
    }
  }
  __syncthreads(); // the ring area is free once every warp has left the
                   // previous task
  // fn's slice to shared memory, as bf16 hi + lo
#pragma unroll
  for (int q = 0; q < FNV; q++) {
    int const idx = threadIdx.x + 256 * q, r = idx / (HC * C),
              k = idx % (HC * C);
    __nv_bfloat16 const hi = __float2bfloat16_rn(fv[q]);
    fh[r * KP + k] = hi;
    fl[r * KP + k] = __float2bfloat16_rn(fv[q] - __bfloat162float(hi));
  }
  STAGE_STAMP_T0(0)
  // 2. the sums: lanes 0 .. 3 of warp w land 8 columns of each of its tokens
  if (lane < C / 8) {
#pragma unroll
    for (int b = 0; b < TB; b++) {
      int const t = warp + 8 * b;
      unsigned char const *row =
          g.rv + SOFF + rg_set() + (size_t)t * H * 2 + (size_t)s * C * 2;
      float sum[8];
      allreduce_land_bf16<S_RANK>(g, row, lane, sum);
#pragma unroll
      for (int k = 0; k < 8; k++) {
        out_s[t * C + 8 * lane + k] =
            __bfloat162float(__float2bfloat16(sum[k])); // the all-reduced bf16
      }
      if constexpr (REARM) {
        allreduce_rearm<S_RANK>(g, row, lane);
      }
    }
  }
  __syncthreads();
  STAGE_STAMP_T0(1)
  // 3. R'
#pragma unroll
  for (int b = 0; b < TB; b++) {
    int const t = warp + 8 * b;
    float const o = out_s[t * C + lane];
#pragma unroll
    for (int j = 0; j < HC; j++) {
      float a = __fmul_rn(comb[b][0][j], rv[b][0]);
#pragma unroll
      for (int i = 1; i < HC; i++) {
        a = __fadd_rn(a, __fmul_rn(comb[b][i][j], rv[b][i]));
      }
      __nv_bfloat16 const bb =
          __float2bfloat16(__fadd_rn(__fmul_rn(post[b][j], o), a));
      rout[((size_t)t * HC + j) * H + c] = bb;
      rb[t * KP + j * C + lane] = bb;
    }
  }
  __syncthreads();
  {
    // 4. the mixes: warp w < MT NT8 has tokens [16 mt, 16 mt + 16) (T = 8:
    // its rows 8 .. 15 repeat 0 .. 7, dropped) x rows [8 nt, 8 nt + 8)
    constexpr int MT = (T + 15) / 16, NT8 = NMIX / 8;
    if (warp < MT * NT8) {
      int const m0 = (warp / NT8) * 16, n0 = (warp % NT8) * 8, g = lane >> 2,
                tq = lane & 3;
      int const t0 = m0 + g, t1 = m0 + g + 8 < T ? m0 + g + 8 : m0 + g;
      float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int k0 = 0; k0 < HC * C; k0 += 16) {
        uint32_t const a[4] = {hcpm_ld2(rb + t0 * KP + k0 + 2 * tq),
                               hcpm_ld2(rb + t1 * KP + k0 + 2 * tq),
                               hcpm_ld2(rb + t0 * KP + k0 + 8 + 2 * tq),
                               hcpm_ld2(rb + t1 * KP + k0 + 8 + 2 * tq)};
        uint32_t const bh[2] = {hcpm_ld2(fh + (n0 + g) * KP + k0 + 2 * tq),
                                hcpm_ld2(fh + (n0 + g) * KP + k0 + 8 + 2 * tq)};
        uint32_t const bl[2] = {hcpm_ld2(fl + (n0 + g) * KP + k0 + 2 * tq),
                                hcpm_ld2(fl + (n0 + g) * KP + k0 + 8 + 2 * tq)};
        hcpm_mma(a, bh, acc);
        hcpm_mma(a, bl, acc);
      }
      *reinterpret_cast<float2 *>(mix + ((size_t)s * T + t0) * HC_MIX_STRIDE +
                                  n0 + 2 * tq) = make_float2(acc[0], acc[1]);
      if (m0 + g + 8 < T) {
        *reinterpret_cast<float2 *>(
            mix + ((size_t)s * T + m0 + g + 8) * HC_MIX_STRIDE + n0 + 2 * tq) =
            make_float2(acc[2], acc[3]);
      }
    }
    // entry NMIX: the sum of squares of the token's HC 32 values, then 0s
    constexpr int TPT = 256 / T, PER = HC * C / TPT;
    int const t = threadIdx.x / TPT, part = threadIdx.x % TPT;
    float q = 0.f;
#pragma unroll
    for (int k = 0; k < PER; k++) {
      float const x = __bfloat162float(rb[t * KP + part * PER + k]);
      q = fmaf(x, x, q);
    }
#pragma unroll
    for (int o = TPT / 2; o > 0; o >>= 1) {
      q += __shfl_xor_sync(0xffffffffu, q, o);
    }
    if (part < HC_MIX_STRIDE - NMIX) {
      mix[((size_t)s * T + t) * HC_MIX_STRIDE + NMIX + part] =
          part == 0 ? q : 0.f;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(done_cnt, 1); // release: R' of these columns, every token
  }
}

// hc_post_mix task x. SEND: the allreduce_send node whose output it adds
// (params {H, 1, FX}); PARAMS {HC, H, NMIX} (hc_pre reads them);
// BUF_SLOTS {the residual streams R, the output R' ([T][HC H] bf16 graph
// tensors), h_post, h_res (fp32 graph tensors), fn [NMIX][HC H] fp32}; its
// buffer: the mixes; its counter: the count of its tasks
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class SEND,
          class... IN>
__device__ __forceinline__ void run_hc_post_mix(Maps const &,
                                                G const &g,
                                                KernelLocals &L,
                                                StaticTask const &tk) {
  constexpr int HC = PARAMS::v[0], H = PARAMS::v[1], NMIX = PARAMS::v[2];
  static_assert(NMIX == HC * (HC + 2),
                "hc_post_mix: params {HC, H, NMIX = HC (HC + 2)}");
  static_assert(SEND::buf >= 0 && SEND::v[0] == H && SEND::v[1] >= 1,
                "hc_post_mix: rows of H sent by an allreduce_send node (a "
                "token's rows of its input added: one row per token sent)");
  static_assert(
      exchange_bytes[SEND::buf] == (size_t)GPUS * T * H * 2,
      "hc_post_mix: the sent sums, [GPUS][T][H] bf16 in the exchange region");
  static_assert(SELF::x * 32 == H && SELF::buf >= 0 && SELF::counter >= 0,
                "hc_post_mix: a task per 32 columns, its mixes and counter");
  hc_post_mix_task<exchange_offset[SEND::buf], H, HC, NMIX>(
      g,
      buf_at<float const>(g, slot_at<BUF_SLOTS, 2>()),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 3>()),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 4>()),
      buf_at<float>(g, SELF::buf),
      cnt_now(g) + SELF::counter,
      tk.x,
      L.sm STAGE_STAMP_ARG(L.stage_stamps));
}

// dynamic shared memory: R' of the task's columns and the landed sums, inside
// the ring area
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_hc_post_mix() {
  static_assert((T * PARAMS::v[0] * 32 + T * 32) * 4 +
                        (T + 2 * PARAMS::v[2]) * hcpm_kp<PARAMS::v[0]>() * 2 <=
                    RING_BYTES,
                "hc_post_mix: its scratch in the ring area");
  return RING_BYTES;
}

} // namespace static_mk
