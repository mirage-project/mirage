// tasks/short_conv_silu.cuh (HIP, gfx950) -- the projection epilogue of a K3
// KDA layer (design: mirage_kda include/mirage/static_megakernel/tasks/
// short_conv_silu.cuh): task x of grid (N / 64, 1, 1) = projection columns
// [64 x, 64 x + 64).
//   z   = the projection's K parts added (a gemm_tile node's K parts in slots:
//         [K parts][T][N] fp32, 0xFF-prefilled, polled), in part order; bf16
//   columns < NCONV (q, k, v: 3 streams of NCONV / 3 channels): a 4-tap causal
//         convolution over the framework's carried window (pool row *slot,
//         3 inputs, oldest first) and the T new inputs, then SiLU, rounded to
//         bf16 (SGLang ROCm), out as fp32 values
//         to [T][NCONV] after the [T][N] bf16 row (as SGLang keeps q / k / v
//         for its recurrence); the window after each token (its 3 newest
//         inputs) to icw at row *icw_row, k / v (bf16) to the rings at the slot;
//         the pool itself is not written (the framework folds the accepted
//         tokens)
//   other columns (g, f_a, beta, padding): bf16(z) into the row
// Then one count on the node's counter (the recurrence waits for all tasks).
// Its output is read only after that count: device-scope stores, no 0xFF.
#pragma once
#include "../core.cuh"

namespace static_mk {

// 4 / 8 bytes at device scope (past the L2: another XCD's CU reads them)
__device__ __forceinline__ void st4_dev(void *p, uint32_t v) {
  asm volatile("global_store_dword %0, %1, off sc1" ::"v"(p), "v"(v)
               : "memory");
}
typedef unsigned scs_v2u __attribute__((ext_vector_type(2)));
__device__ __forceinline__ void st8x_dev(void *p, uint2 v) {
  scs_v2u const w = {v.x, v.y};
  asm volatile("global_store_dwordx2 %0, %1, off sc1" ::"v"(p), "v"(w)
               : "memory");
}

template <int NPART, int N, int NCONV>
__device__ __forceinline__ void short_conv_silu_task(KernelLocals &L,
                                                     float const *zpart,
                                                     float const *conv_w,
                                                     __hip_bfloat16 const *pool,
                                                     __hip_bfloat16 *icw,
                                                     __hip_bfloat16 *ring_k,
                                                     __hip_bfloat16 *ring_v,
                                                     int const *info,
                                                     int const *slot_p,
                                                     int const *row_p,
                                                     __hip_bfloat16 *out,
                                                     uint32_t *cnt,
                                                     int x) {
  static_assert(N % 64 == 0 && NCONV % 384 == 0 && NCONV <= N,
                "short_conv_silu: 64-column tasks, 3 streams of whole heads");
  constexpr int HD = NCONV / 3;
  int const tid = task_tid_all(), c0 = x * 64;
  bool const conv = c0 < NCONV;
  // the framework's scalars and this channel's taps / window: no input needed,
  // loaded before the poll
  // (slot, icw_row: the framework's index tensors, refreshed in place before
  // each step: a graph replay reads the step's values)
  int const ring_len = info[3], steps = info[4], slot = slot_p[0],
            icw_row = row_p[0];
  float w[4] = {0.f, 0.f, 0.f, 0.f}, win[3] = {0.f, 0.f, 0.f};
  if (conv && tid < 64) {
    float4 const w4 =
        *reinterpret_cast<float4 const *>(conv_w + (size_t)(c0 + tid) * 4);
    w[0] = w4.x;
    w[1] = w4.y;
    w[2] = w4.z;
    w[3] = w4.w;
#pragma unroll
    for (int j = 0; j < 3; j++) {
      win[j] = __bfloat162float(
          pool[((size_t)slot * 3 + j) * NCONV + c0 + tid]);
    }
  }
  float *zs = reinterpret_cast<float *>(L.sm); // [T][64] the bf16 inputs
  __syncthreads(); // the LDS may still be read by the previous task
  if (tid < 128) { // token tid / 16, columns 4 (tid % 16) .. + 3
    int const t = tid >> 4, c = c0 + 4 * (tid & 15);
    float4 zp[NPART];
    bool all;
    do {
#pragma unroll
      for (int p = 0; p < NPART; p++) { // all parts' loads in flight at once
        // (device-scope buffer loads the compiler tracks: all parts in flight;
        // ldf4_dev waits for each load)
        zp[p] = as_float4(ld16_at<AUX_DEV>(zpart, (uint32_t)((((size_t)p * T + t) * N + c) * 4)));
      }
      all = true;
#pragma unroll
      for (int p = 0; p < NPART; p++) {
        all = all && validf4(zp[p]);
      }
      if (!all) {
        tail_pause();
      }
    } while (!all);
    float zv[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int p = 0; p < NPART; p++) { // part order
      zv[0] += zp[p].x;
      zv[1] += zp[p].y;
      zv[2] += zp[p].z;
      zv[3] += zp[p].w;
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
      zv[j] = round_bf16(zv[j]);
      zs[t * 64 + 4 * (tid & 15) + j] = zv[j];
    }
    if (!conv) {
      st8x_dev(out + (size_t)t * N + c,
               make_uint2(pack_bf16x2(zv[0], zv[1]), pack_bf16x2(zv[2], zv[3])));
    }
  }
  __syncthreads();
  if (conv && tid < 64) { // channel c0 + tid
    float win11[3 + T];
    win11[0] = win[0];
    win11[1] = win[1];
    win11[2] = win[2];
#pragma unroll
    for (int m = 0; m < T; m++) {
      win11[3 + m] = zs[m * 64 + tid];
    }
    int const cc = c0 + tid;
    float *const qkv = reinterpret_cast<float *>(out + (size_t)T * N);
    int const stream = cc / HD, ch = cc % HD; // 0 q, 1 k, 2 v
    __hip_bfloat16 *const ring = stream == 1 ? ring_k : ring_v;
#pragma unroll
    for (int m = 0; m < T; m++) {
      float acc = 0.f;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        acc = fmaf(win11[m + j], w[j], acc);
      }
      // bf16 (SGLang ROCm's causal_conv1d writes the conv + SiLU output in
      // bf16; the recurrence reads that)
      float const sy = round_bf16(acc / (1.0f + __expf(-acc)));
      st4_dev(qkv + (size_t)m * NCONV + cc, __float_as_uint(sy));
      if (stream > 0) { // the ring entry of token m (bf16, after SiLU)
        ring[(((size_t)slot * (HD / 128) + ch / 128) * ring_len + m) * 128 +
             ch % 128] = __float2bfloat16(sy);
      }
#pragma unroll
      for (int j = 0; j < 3; j++) { // the window after token m
        icw[(((size_t)icw_row * steps + m) * 3 + j) * NCONV + cc] =
            __float2bfloat16(win11[m + 1 + j]);
      }
    }
  }
  task_post(cnt, 1);
}

// task x: Z = the gemm_tile node whose K parts it adds (grid y = the parts, N =
// its params[2]); params = [NCONV]; buffer slots = conv weights [NCONV][4] fp32,
// the window pool, icw, ring_k, ring_v, info, slot, icw_row
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class Z,
          class... IN>
__device__ __forceinline__ void run_short_conv_silu(Maps const &,
                                                    G const &g,
                                                    KernelLocals &L,
                                                    StaticTask const &tk) {
  static_assert(Z::buf >= 0 && Z::v[0] == COMBINE_SLOTS,
                "short_conv_silu: a gemm_tile node's K parts in slots");
  static_assert(SELF::buf >= 0 && SELF::counter >= 0 && SELF::x * 64 == Z::v[2],
                "short_conv_silu: one task per 64 columns, its output and a counter");
  short_conv_silu_task<Z::y, Z::v[2], PARAMS::v[0]>(
      L,
      buf_at<float const>(g, Z::buf),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<__hip_bfloat16>(g, slot_at<BUF_SLOTS, 2>()),
      buf_at<__hip_bfloat16>(g, slot_at<BUF_SLOTS, 3>()),
      buf_at<__hip_bfloat16>(g, slot_at<BUF_SLOTS, 4>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 5>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 6>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 7>()),
      buf_at<__hip_bfloat16>(g, SELF::buf),
      cnt_of(g) + SELF::counter,
      tk.x);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_short_conv_silu() {
  return T * 64 * 4;
}

} // namespace static_mk
