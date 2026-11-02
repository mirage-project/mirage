// tasks/short_conv_decode.cuh (HIP, gfx950) -- the projection epilogue of a K3 KDA layer in plain decode: T requests, one
// new token each (row t = request t; SGLang's causal_conv1d_update, amd/notes/sglang_v0521_decode_contract.md B.3). Task
// (x, y) of grid (N / 64, Y, 1) = projection columns [64 x, 64 x + 64) of the tokens [TS y, TS y + TS) (TS = T / Y):
//   z   = the projection's K parts added (a gemm_tile node's K parts in slots: [K parts][T][N] fp32, 0xFF-prefilled,
//         polled), in part order; bf16
//   columns < NCONV (q, k, v): per request t with state slot s = slot[t] >= 0, the 4-tap causal convolution over its
//         window conv[s] = (c0, c1, c2) (bf16 [slots][3][NCONV], oldest first) and z[t], then SiLU, rounded to bf16, out as
//         fp32 to [T][NCONV] after the [T][N] bf16 row; then conv[s] = (c1, c2, z[t]) (in place: the state SGLang's next
//         step reads). A padded row (s < 0): no state read or write, its output zeros.
//   other columns (g, f_a, beta, padding): bf16(z) into the row
// Then one count on the node's counter. Its output is read only after that count: device-scope stores, no 0xFF.
#pragma once
#include "../core.cuh"
#include "short_conv_silu.cuh" // st4_dev, st8x_dev

namespace static_mk {

template <int NPART, int N, int NCONV>
__device__ __forceinline__ void short_conv_decode_task(KernelLocals &L,
                                                       float const *zpart,
                                                       float const *conv_w,
                                                       __hip_bfloat16 *pool,
                                                       int const *slots,
                                                       __hip_bfloat16 *out,
                                                       uint32_t *cnt,
                                                       int x,
                                                       int t0, // this task's tokens [t0, t0 + nt)
                                                       int nt) {
  static_assert(N % 64 == 0 && NCONV % 384 == 0 && NCONV <= N,
                "short_conv_decode: 64-column tasks, 3 streams of whole heads");
  int const tid = task_tid_all(), c0 = x * 64;
  bool const conv = c0 < NCONV;
  float *zs = reinterpret_cast<float *>(L.sm); // [T][64] the bf16 inputs
  __syncthreads(); // the LDS may still be read by the previous task
  // 1. z: item v = token t0 + v / 16, columns 4 (v % 16) .. + 3
  for (int v = tid; v < nt * 16; v += 256) {
    int const t = t0 + (v >> 4), c = c0 + 4 * (v & 15);
    float4 zp[NPART];
    bool all;
    do {
#pragma unroll
      for (int p = 0; p < NPART; p++) { // all parts' loads in flight at once
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
      zs[t * 64 + 4 * (v & 15) + j] = zv[j];
    }
    if (!conv) {
      st8x_dev(out + (size_t)t * N + c, make_uint2(pack_bf16x2(zv[0], zv[1]), pack_bf16x2(zv[2], zv[3])));
    }
  }
  __syncthreads();
  // 2. the conv: item v = token v / 64, channel c0 + v % 64
  if (conv) {
    float *const qkv = reinterpret_cast<float *>(out + (size_t)T * N);
    for (int v = tid; v < nt * 64; v += 256) {
      int const t = t0 + (v >> 6), cc = c0 + (v & 63);
      int const s = slots[t];
      float y = 0.f;
      if (s >= 0) {
        float4 const w4 = *reinterpret_cast<float4 const *>(conv_w + (size_t)cc * 4);
        __hip_bfloat16 *const win = pool + (size_t)s * 3 * NCONV + cc;
        float const a0 = __bfloat162float(win[0]), a1 = __bfloat162float(win[NCONV]), a2 = __bfloat162float(win[2 * NCONV]);
        float const xn = zs[t * 64 + (v & 63)];
        float acc = 0.f;
        acc = fmaf(a0, w4.x, acc);
        acc = fmaf(a1, w4.y, acc);
        acc = fmaf(a2, w4.z, acc);
        acc = fmaf(xn, w4.w, acc);
        // bf16 (SGLang ROCm's causal_conv1d writes the conv + SiLU output in bf16; the recurrence reads that)
        y = round_bf16(acc / (1.0f + __expf(-acc)));
        win[0] = __float2bfloat16(a1); // the window after the token: (c1, c2, x)
        win[NCONV] = __float2bfloat16(a2);
        win[2 * NCONV] = __float2bfloat16(xn);
      }
      st4_dev(qkv + (size_t)t * NCONV + cc, __float_as_uint(y));
    }
  }
  task_post(cnt, 1);
}

// task x: Z = the gemm_tile node whose K parts it adds (grid y = the parts, N = its params[2]); params = [NCONV]; buffer
// slots = conv weights [NCONV][4] fp32, the window pool [slots][3][NCONV] bf16 (read and written), the per-request slots
// int32 [T]
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class Z,
          class... IN>
__device__ __forceinline__ void run_short_conv_decode(Maps const &,
                                                      G const &g,
                                                      KernelLocals &L,
                                                      StaticTask const &tk) {
  static_assert(Z::buf >= 0 && Z::v[0] == COMBINE_SLOTS, "short_conv_decode: a gemm_tile node's K parts in slots");
  static_assert(SELF::buf >= 0 && SELF::counter >= 0 && (SELF::x * 64 == Z::v[2] || SELF::x * 64 == PARAMS::v[0]),
                "short_conv_decode: one task per 64 columns (all, or the conv's only) and token group, its output and a "
                "counter");
  constexpr int TS = (T + SELF::y - 1) / SELF::y; // tokens per task
  int const t0 = tk.y * TS, nt = T - t0 < TS ? T - t0 : TS;
  short_conv_decode_task<Z::y, Z::v[2], PARAMS::v[0]>(
      L,
      buf_at<float const>(g, Z::buf),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<__hip_bfloat16>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 2>()),
      buf_at<__hip_bfloat16>(g, SELF::buf),
      cnt_of(g) + SELF::counter,
      tk.x,
      t0,
      nt);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_short_conv_decode() {
  return T * 64 * 4;
}

} // namespace static_mk
