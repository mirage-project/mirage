// tasks/q_norm.cuh -- DeepSeek-V4.1's q_norm and the act_quant of its output
// (model.py L771: qr = q_norm(wq_a(x)); wq_b's linear quantizes qr as
// act_quant does): task t = token t, grid (T, 1, 1); K = 1280 (q_lora_rank)
// columns over the first K / 8 threads, thread i columns 8 i .. 8 i + 7 (a
// warp 256 columns, 4 threads a 32-column group; the other warps only take
// part in the barriers). Add wq_a's P K parts in part order -> bf16 (wq_a's
// bf16 output) -> RMSNorm (rms_norm_parts8): qr = bf16(x * rsqrt(mean(x^2) +
// eps) * gamma) -> act_quant's formula per 32 columns into the node's buffer,
// laid out as act_quant's (x_q, then the scale chunks; every byte written each
// launch); then the node's counter += 1 (wq_b's tasks wait for T).
#pragma once
#include "../core.cuh"
#include "act_quant.cuh"

namespace static_mk {

// the RMSNorm of token t's row of K columns as model.py's RMSNorm, from a
// gemm_fp8_tile node's P K parts (fp32, its buffer in slots [P][T][K], 0xFF
// before the launch: polled): the parts added in part order -> bf16 (the
// GEMM's bf16 output) -> y = bf16(x * rsqrt(mean(x^2) + eps) * gamma), fp32
// inside. Thread i < K / 8: columns 8 i .. 8 i + 7 into y (a warp 256 columns);
// returns false for the other threads (whole warps). Every thread of the CTA
// calls it (a barrier). dry: one round of loads, not waited for (a dry pass).
// mean(x^2): each square rounded, then added in a fixed order (a thread's 8
// columns in order, the warp's 32 by a butterfly, the warps in order), not
// torch's, so a few y in 10^5 are 1 bf16 step from the official's.
template <int K, int P, int EPS_BITS>
__device__ __forceinline__ bool
    rms_norm_parts8(float const *part,
                    __nv_bfloat16 const *gamma,
                    int t,
                    float (&y)[8],
                    bool dry = false STAGE_STAMP_PARAM) {
  constexpr int NW = K / 256; // the warps with columns
  static_assert(K % 256 == 0 && NW >= 1 && NW <= 8,
                "rms_norm_parts8: 8 columns per thread, 256 per warp, whole "
                "warps of the 256 threads");
  int const w = threadIdx.x >> 5, l = threadIdx.x & 31, c = 8 * threadIdx.x;
  bool const on = w < NW;    // the same for a whole warp
  __shared__ float red_q[8]; // the warps' sums of squares
  float x[8];
  uint4 gb;
  if (on) {
    // gamma first: its load overlaps the wait for the parts (after the barrier
    // below it could not start before the sum of squares)
    gb = *reinterpret_cast<uint4 const *>(gamma + c);
    add_parts8<K, P>(part, t, c, x, dry);
    STAGE_STAMP(true)
    float ss = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) {
      x[k] = __bfloat162float(__float2bfloat16_rn(x[k])); // the GEMM's output
      ss = __fadd_rn(ss, __fmul_rn(x[k], x[k])); // no fma: the squares rounded
    }
    for (int d = 16; d > 0; d >>= 1) {
      ss = __fadd_rn(ss, __shfl_xor_sync(0xffffffffu, ss, d));
    }
    if (l == 0) {
      red_q[w] = ss;
    }
  }
  __syncthreads();
  if (on) {
    float tot = red_q[0];
#pragma unroll
    for (int i = 1; i < NW; i++) {
      tot = __fadd_rn(tot, red_q[i]);
    }
    float const r =
        rsqrtf(__fadd_rn(__fmul_rn(tot, 1.0f / K), __int_as_float(EPS_BITS)));
    __nv_bfloat162 const *g2 = reinterpret_cast<__nv_bfloat162 const *>(&gb);
#pragma unroll
    for (int k = 0; k < 4; k++) {
      float2 const gm = __bfloat1622float2(g2[k]);
      y[2 * k] = __bfloat162float(
          __float2bfloat16_rn(__fmul_rn(__fmul_rn(x[2 * k], r), gm.x)));
      y[2 * k + 1] = __bfloat162float(
          __float2bfloat16_rn(__fmul_rn(__fmul_rn(x[2 * k + 1], r), gm.y)));
    }
  }
  return on;
}

// warm = false (this node's first task on the SM): first a dry pass,
// the same code with one round of loads and no stores, while wq_a's parts are
// on their way; one copy of the code (unroll 1)
// timing build: stamp [0] the dry pass done, [1] thread 0's parts landed
template <int K, int P, int EPS_BITS>
__device__ __forceinline__ void q_norm_task(float const *part,
                                            __nv_bfloat16 const *gamma,
                                            unsigned char *out,
                                            uint32_t *done_cnt,
                                            int t,
                                            bool warm STAGE_STAMP_PARAM) {
  int const l = threadIdx.x & 31, c = 8 * threadIdx.x;
#pragma unroll 1
  for (int pass = warm ? 1 : 0; pass < 2; pass++) {
    bool const real = pass == 1;
    float y[8]; // qr
    if (threadIdx.x == 0 && real) {
      STAGE_STAMP_AT(0)
    }
    if (rms_norm_parts8<K, P, EPS_BITS>(
            part,
            gamma,
            t,
            y,
            !real STAGE_STAMP_ARG(real && threadIdx.x == 0 ? stamp + 1
                                                           : nullptr))) {
      float amax = 0.f;
#pragma unroll
      for (int k = 0; k < 8; k++) {
        amax = fmaxf(amax, fabsf(y[k]));
      }
      for (int d = 1; d < 4; d <<= 1) { // over the 4 threads of the group
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
      }
      int const ex = xq_exponent(amax);
      uint2 const q = make_uint2(xq_pack4(y, ex), xq_pack4(y + 4, ex));
      uint32_t const s4 =
          xq_scale_word<16>(ex); // 16 threads = a 128-column block
      if (real) {
        *reinterpret_cast<uint2 *>(out + (size_t)t * K + c) = q;
        if ((l & 15) < 4) {
          xq_scale_line(out, K, c / 128, t, l & 15, s4);
        }
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(done_cnt, 1); // release: the CTA's stores before (the barrier)
  }
  // the parts re-armed after the count (its release does not wait for these
  // stores; the next writers are the next launch's)
  if (c < K) {
    rearm_parts8<K, P>(part, t, c);
  }
}

// q_norm task t = token t, grid (T, 1, 1); QA: wq_a's node (a gemm_fp8_tile in
// slots: QA::y K parts). PARAMS {K, eps (fp32 bits)}; BUF_SLOTS {gamma (a graph
// tensor, [K] bf16)}; warm: the dry pass (q_norm_task)
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class QA,
          class... REST>
__device__ __forceinline__ void run_q_norm(Maps const &,
                                           G const &g,
                                           KernelLocals &L,
                                           StaticTask const &tk,
                                           bool warm = true) {
  constexpr int K = PARAMS::v[0];
  static_assert(SELF::x == T && SELF::y == 1 && SELF::z == 1,
                "q_norm: one task per token");
  static_assert(SELF::buf >= 0 && exchange_bytes[SELF::buf] == 0 &&
                    SELF::counter >= 0,
                "q_norm: its output in this GPU's memory, a counter of its "
                "tasks");
  static_assert(QA::buf >= 0 && QA::v[0] == COMBINE_SLOTS && QA::v[2] == K &&
                    QA::y >= 1 && QA::y <= 10,
                "q_norm: wq_a's K parts in slots, at most 10 (in registers, 8 "
                "values each)");
  q_norm_task<K, QA::y, PARAMS::v[1]>(
      buf_at<float const>(g, QA::buf),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<unsigned char>(g, SELF::buf),
      g.cnt + SELF::counter,
      tk.x,
      warm STAGE_STAMP_ARG(L.stage_stamps));
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_q_norm() {
  return 0;
}

} // namespace static_mk
