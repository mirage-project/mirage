// tasks/short_conv_silu.cuh -- the projection epilogue of a KDA layer (Kimi
// K3; the projection GEMM's epilogue with the short convolution folded in):
// task x of grid (N / 64, 1, 1) = projection columns [64 x, 64 x + 64).
//   z     = the projection's K parts added (a gemm_tile node's output in slots:
//           [K parts][T][N] fp32, 0xFF-prefilled), in part order
//   y     = bf16(z) (-0 stored as +0: bf16 0x8000 marks "not written" for
//           readers that poll)
//   for columns < NCONV (q, k, v: 3 streams of HD channels): a 4-tap causal
//   convolution over the 3 carried inputs (the window state, slots 1..3) and
//   the T new inputs, then SiLU: out[t] = bf16(silu(sum_j w[j] win[t + j]));
//   the last 3 inputs become the new carried window (written in place)
//   the other columns (g, f_a, beta): out = y
// Output: [T][N] bf16 (the node's buffer, the projection's layout), then one
// count on the node's counter. Shared memory after RING_BYTES (2 KB).
// SERVE (params[1], serving builds; SGLang's KDA target-verify contract): the
// window is read from the framework's pool [slots][3][NCONV] bf16 at *slot
// (oldest first) and not written back; per token m the window after it (the 3
// newest inputs) goes to icw [rows][ICW steps][3][NCONV] at *icw_row, and the k
// / v channels' outputs (pre-L2-norm, bf16) to the rings ring_k / ring_v
// [slots][heads][L][128] at position m (L, ICW steps: info[3], info[4]) QKV32
// (params[2]; armed output only): q, k, v after the conv + SiLU go out in fp32,
// as SGLang's kda_decode_mtp keeps them for its recurrence, to [T][NCONV]
// floats after the [T][N] bf16 row (the node's buffer is that much larger);
// their bf16 columns of the row are not written (the rings and windows stay
// bf16)
#pragma once
#include "../core.cuh"

// CONV_DIAG_STAMPS (timing builds, diagnosis only): 1 = stage 0 every input
// seen (after the poll's barrier), stage 1 the outputs written (before the
// count); 2 = stage 0 the conv weights / window in registers (thread 0, waited
// for right after their loads), stage 1 every input seen

namespace static_mk {

struct ConvServe { // SERVE: the framework's buffers (see above)
  int const *slot, *icw_row, *info;
  __nv_bfloat16 *icw, *ring_k, *ring_v;
};

template <int NSPLIT, int N, int NCONV, int SERVE = 0, int QKV32 = 0>
__device__ __forceinline__ void short_conv_silu_task(KernelLocals &L,
                                                     float const *zpart,
                                                     float const *conv_w,
                                                     __nv_bfloat16 *conv_s,
                                                     __nv_bfloat16 *out,
                                                     uint32_t *cnt,
                                                     int x,
                                                     ConvServe const sv = {}) {
  static_assert(N % 64 == 0 && NCONV % 64 == 0 && NCONV <= N,
                "short_conv_silu: 64-column tasks");
  int const tid = threadIdx.x, c0 = x * 64;
  float *zs = reinterpret_cast<float *>(
      L.sm + RING_BYTES); // [T][64] bf16-rounded inputs
  // the conv channel's weights and carried window (no inputs needed: loaded
  // first)
  float w[4] = {0.f, 0.f, 0.f, 0.f}, win[4] = {0.f, 0.f, 0.f, 0.f};
  bool const conv = c0 < NCONV;
  int slot = 0, ring_len = 0, icw_row = 0, steps = 0;
  if constexpr (SERVE) { // the framework's scalars, read before the poll (their
                         // latency hidden behind it, not after it)
    slot = *sv.slot;
    ring_len = sv.info[3];
    icw_row = *sv.icw_row;
    steps = sv.info[4];
  }
  if (conv && tid < 64) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
      w[j] = conv_w[(size_t)(c0 + tid) * 4 + j];
      if constexpr (SERVE) { // the pool's window [slot][3][NCONV], oldest first
                             // -> win[1..3]
        win[j] =
            j == 0
                ? 0.f
                : __bfloat162float(
                      conv_s[((size_t)slot * 3 + (j - 1)) * NCONV + c0 + tid]);
      } else {
        win[j] = __bfloat162float(conv_s[(size_t)(c0 + tid) * 4 + j]);
      }
    }
  }
  __syncthreads(); // zs may be another task's shared memory until here
  if (tid < 128) { // token tid >> 4, columns 4 (tid & 15) .. + 3
    int const t = tid >> 4, c = c0 + 4 * (tid & 15);
    float4 zp[NSPLIT];
    bool all;
    do {
#pragma unroll
      for (int p = 0; p < NSPLIT;
           p++) { // no memory clobber: the parts' loads of one round in flight
                  // together (ldf4_relaxed
        // orders them: one round trip each; 10-10 inputs seen 2.1 us after the
        // projection's last task)
        float4 v;
        asm volatile("ld.relaxed.gpu.global.v4.f32 {%0,%1,%2,%3}, [%4];"
                     : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w)
                     : "l"(zpart + ((size_t)p * T + t) * N + c));
        zp[p] = v;
      }
      all = true;
#pragma unroll
      for (int p = 0; p < NSPLIT; p++) {
        all = all && validf4(zp[p]);
      }
    } while (!all);
    float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
    for (int p = 0; p < NSPLIT; p++) {
      z.x += zp[p].x;
      z.y += zp[p].y;
      z.z += zp[p].z;
      z.w += zp[p].w;
    }
    float const zv[4] = {z.x, z.y, z.z, z.w};
#pragma unroll
    for (int j = 0; j < 4; j++) {
      __nv_bfloat16 h = __float2bfloat16(zv[j]);
      if (__bfloat16_as_ushort(h) == 0x8000u) {
        h = __float2bfloat16(0.f);
      }
      zs[t * 64 + 4 * (tid & 15) + j] = __bfloat162float(h);
      if (!conv) {
        out[(size_t)t * N + c + j] = h;
      }
    }
    if constexpr (REARM) { // the one reader of these parts
#pragma unroll
      for (int p = 0; p < NSPLIT; p++) {
        rearm16(reinterpret_cast<unsigned char *>(const_cast<float *>(zpart) +
                                                  ((size_t)p * T + t) * N + c));
      }
    }
  }
  __syncthreads();
  if (conv && tid < 64) { // channel c0 + tid: the window (3 carried + T new
                          // inputs), the conv, SiLU
    float win11[3 + T];
    win11[0] = win[1];
    win11[1] = win[2];
    win11[2] = win[3];
#pragma unroll
    for (int m = 0; m < T; m++) {
      win11[3 + m] = zs[m * 64 + tid];
    }
    __nv_bfloat16 cys[T];
#pragma unroll
    for (int m = 0; m < T; m++) {
      float acc = 0.f;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        acc = fmaf(win11[m + j], w[j], acc);
      }
      float const sy = __fdividef(acc, 1.0f + __expf(-acc));
      __nv_bfloat16 cy = __float2bfloat16(sy);
      if (__bfloat16_as_ushort(cy) == 0x8000u) {
        cy = __float2bfloat16(0.f);
      }
      if constexpr (QKV32) {
        reinterpret_cast<float *>(
            out + (size_t)T * N)[(size_t)m * NCONV + c0 + tid] = sy;
      } else {
        out[(size_t)m * N + c0 + tid] = cy;
      }
      cys[m] = cy;
    }
    if constexpr (SERVE) { // k / v channels: the ring entry of each token
                           // (heads x 128 channels per stream), after every
      constexpr int HD =
          NCONV /
          3; // output store (the recurrence polls the outputs: they go first)
      int const cc = c0 + tid;
      if (cc >= HD) {
        int const cv = cc - HD, stream = cv / HD,
                  ch = cv % HD; // stream 0: k, 1: v
        __nv_bfloat16 *ring = stream == 0 ? sv.ring_k : sv.ring_v;
#pragma unroll
        for (int m = 0; m < T; m++) {
          ring[(((size_t)slot * (HD / 128) + ch / 128) * ring_len + m) * 128 +
               ch % 128] = cys[m];
        }
      }
    }
    if constexpr (SERVE) { // per token m the window after it (inputs m + 1 .. m
                           // + 3 of win11) -> icw; the pool unchanged
      int const row = icw_row;
#pragma unroll
      for (int m = 0; m < T; m++) {
#pragma unroll
        for (int j = 0; j < 3; j++) {
          sv.icw[(((size_t)row * steps + m) * 3 + j) * NCONV + c0 + tid] =
              __float2bfloat16(win11[m + 1 + j]);
        }
      }
    } else {
      __nv_bfloat16 *s =
          conv_s +
          (size_t)(c0 + tid) *
              4; // the new carried window: slots 1..3 = the last 3 inputs
      s[0] = __float2bfloat16(0.f);
      s[1] = __float2bfloat16(win11[T]);
      s[2] = __float2bfloat16(win11[T + 1]);
      s[3] = __float2bfloat16(win11[T + 2]);
    }
  }
  __syncthreads();
  if (tid == 0) {
    cnt_add(cnt, 1); // the recurrence waits for every task
  }
}

// task x: Z = the gemm_tile node whose K parts it adds (grid y = the parts, N =
// its params[2]); params = [NCONV, SERVE (optional), QKV32 (optional)]; buffer
// slots = conv weights [NCONV][4] fp32, the window state [NCONV][4] bf16 (read
// and written); SERVE: the pool's windows [slots][3][NCONV], then slot, icw,
// icw_row, ring_k, ring_v, info
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
  static_assert(
      SELF::buf >= 0 && SELF::counter >= 0 && SELF::x * 64 == Z::v[2],
      "short_conv_silu: one task per 64 columns, its output and a counter");
  constexpr int SERVE = param_or<PARAMS, 1>(), QKV32 = param_or<PARAMS, 2>();
  ConvServe sv{};
  if constexpr (SERVE) {
    sv.slot = buf_at<int const>(g, slot_at<BUF_SLOTS, 2>());
    sv.icw = buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 3>());
    sv.icw_row = buf_at<int const>(g, slot_at<BUF_SLOTS, 4>());
    sv.ring_k = buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 5>());
    sv.ring_v = buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 6>());
    sv.info = buf_at<int const>(g, slot_at<BUF_SLOTS, 7>());
  }
  short_conv_silu_task<Z::y, Z::v[2], PARAMS::v[0], SERVE, QKV32>(
      L,
      buf_at<float const>(g, Z::buf),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<__nv_bfloat16>(g, SELF::buf),
      g.cnt + SELF::counter,
      tk.x,
      sv);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_short_conv_silu() {
  return RING_BYTES + T * 64 * 4;
}

} // namespace static_mk
