// tasks/group_quant.cuh -- Convert a bf16 activation to e4m3 with fp32 scales.
// scaled_moe_experts and scaled_gemm_tile read this output.
// A K tile is 128 activation columns. z_q is the quantized activation.
//   input    bf16 [T][L]: a graph input (x; each GPU has all of x) or the
//            output buffer of a node (h_s of silu_and_mul)
//   task kt  one K tile: columns [128 kt, 128 kt + 128)
//   scale    for each token, s = max(amax of the 128 values, 1e-10) / 448.
//            Each value becomes e4m3(value / s). SGLang and vLLM use the same
//            method (per_token_group_quant_fp8, group size 128).
//   output   this GPU's copy in the exchange region: z_q [T][L] e4m3, then the
//            scales [L / 128][T] fp32 (zq32_bytes)
// Two readers use the output:
//   1. scaled_moe_experts polls each 16-byte unit for a change from 0xFF.
//      Each unit must come from one store. Collect four lanes of bytes
//      before the store. Collect the T scales in shared memory.
//   2. scaled_gemm_tile reads the output through tensor maps.
//      Each task adds one to the completion counter after all its stores.
#pragma once
#include "../core.cuh"

namespace static_mk {

// the output size for width L: z_q bytes, then [L / 128][T] fp32 scales
__host__ __device__ constexpr size_t zq32_bytes(int L) {
  return (size_t)T * L + (size_t)(L / 128) * T * 4;
}

// ZOFF: the output's offset in the exchange region
template <size_t ZOFF, int L>
__device__ __forceinline__ void group_quant_task(G const &g,
                                                 __nv_bfloat16 const *x,
                                                 uint32_t *done_cnt,
                                                 int kt) {
  __shared__ __align__(16) float scale_s[T];
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  // The previous task finished its scale_s reads before its final barrier.
  if (warp < 4) { // warp w: tokens 2w and 2w + 1. A lane: 4 columns.
#pragma unroll
    for (int i = 0; i < 2; i++) {
      int const t = 2 * warp + i;
      uint2 raw; // Bypass L1 because another SM can produce x.
      asm volatile("ld.relaxed.gpu.global.v2.u32 {%0, %1}, [%2];"
                   : "=r"(raw.x), "=r"(raw.y)
                   : "l"(x + (size_t)t * L + kt * 128 + 4 * lane)
                   : "memory");
      float2 const a = __bfloat1622float2(
                       *reinterpret_cast<__nv_bfloat162 const *>(&raw.x)),
                   b = __bfloat1622float2(
                       *reinterpret_cast<__nv_bfloat162 const *>(&raw.y));
      float amax =
          fmaxf(fmaxf(fabsf(a.x), fabsf(a.y)), fmaxf(fabsf(b.x), fabsf(b.y)));
      for (int d = 16; d > 0; d >>= 1) { // amax of the 32 lanes
        amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
      }
      float const s = fmaxf(amax, 1.0e-10f) * (1.0f / 448.0f);
      uint32_t const q = (uint32_t)cvt_e4m3(__fdiv_rn(a.x, s)) |
                         ((uint32_t)cvt_e4m3(__fdiv_rn(a.y, s)) << 8) |
                         ((uint32_t)cvt_e4m3(__fdiv_rn(b.x, s)) << 16) |
                         ((uint32_t)cvt_e4m3(__fdiv_rn(b.y, s)) << 24);
      if (lane == 0) {
        scale_s[t] = s;
      }
      uint32_t const q1 = __shfl_down_sync(0xffffffffu, q, 1),
                     q2 = __shfl_down_sync(0xffffffffu, q, 2),
                     q3 = __shfl_down_sync(0xffffffffu, q, 3);
      if ((lane & 3) == 0) {
        *reinterpret_cast<uint4 *>(g.rv + ZOFF + (size_t)t * L + kt * 128 +
                                   4 * lane + rg_set()) =
            make_uint4(q, q1, q2, q3);
      }
    }
  }
  __syncthreads();
  if (threadIdx.x < T * 4 / 16) { // the T scales of this K tile: two stores
    *reinterpret_cast<uint4 *>(
        g.rv + ZOFF + (size_t)T * L + (size_t)kt * T * 4 + threadIdx.x * 16 +
        rg_set()) = reinterpret_cast<uint4 const *>(scale_s)[threadIdx.x];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(done_cnt, 1); // Publish all threads' stores after the CTA barrier.
  }
}

// group_quant task kt (grid x): K tile kt of the input. PARAMS {L (the width
// of the input)}. BUF_SLOTS {the input (a graph tensor, [T][L] bf16), or -1
// when X writes the input}. X: the node that writes the input (none: a graph
// input). The task waits until X has counted all its tasks.
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class X,
          class... IN>
__device__ __forceinline__ void run_group_quant(Maps const &,
                                                G const &g,
                                                KernelLocals &,
                                                StaticTask const &tk) {
  constexpr int L = PARAMS::v[0], XS = slot_at<BUF_SLOTS, 0>();
  static_assert(L % 128 == 0 && SELF::x * 128 == L,
                "group_quant: one task for each 128 columns of x");
  static_assert(SELF::buf >= 0 && exchange_bytes[SELF::buf] == zq32_bytes(L),
                "group_quant: the output z_q is in the exchange region");
  static_assert(SELF::counter >= 0, "group_quant: a count of its tasks");
  static_assert(
      (X::x > 0) == (XS < 0),
      "group_quant: x is a graph tensor (its slot) or a node's output");
  if constexpr (X::x > 0) {
    static_assert(X::buf >= 0 && X::counter >= 0,
                  "group_quant: x's producer counts its tasks");
    if (threadIdx.x == 0) {
      cnt_wait(g.cnt + X::counter, X::x * X::y * X::z);
    }
    __syncthreads();
  }
  group_quant_task<exchange_offset[SELF::buf], L>(
      g,
      buf_at<__nv_bfloat16 const>(g, X::x > 0 ? X::buf : XS),
      g.cnt + SELF::counter,
      tk.x);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_group_quant() {
  return 0;
}

} // namespace static_mk
