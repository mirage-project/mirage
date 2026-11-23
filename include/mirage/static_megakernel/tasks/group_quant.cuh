// tasks/group_quant.cuh -- Convert a bf16 activation to e4m3 with fp32 scales.
// scaled_moe_experts and scaled_gemm_tile read this output.
// A K tile is 128 activation columns. z_q is the quantized activation.
//   input    bf16 [T][L]: a graph input (x; each GPU has all of x) or the
//            output buffer of a node (h_s of silu_and_mul)
//   task kt  one K tile: columns [128 kt, 128 kt + 128), of all tokens; or
//            (grid y NZ > 1) task (kt, z): of token tile z of NZ
//            (group_quant_tile_task)
//   scale    for each token, s = max(amax of the 128 values, 1e-10) / 448.
//            Each value becomes e4m3(value / s). SGLang and vLLM use the same
//            method (per_token_group_quant_fp8, group size 128). UE8M0
//            (params[1]): s rounded up to a power of two (group_scale), as
//            SGLang's scale_ue8m0 for DeepGEMM on Blackwell; the scales keep
//            their fp32 layout.
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

// the scale of a 128-value group with amax `amax`: max(amax, 1e-10) / 448, or
// (UE8M0) the power of two at or above max(amax, 1e-10) / 448 (DeepGEMM's
// ceil_to_ue8m0 of the quotient)
template <bool UE8M0>
__device__ __forceinline__ float group_scale(float amax) {
  if constexpr (UE8M0) {
    uint32_t const b =
        __float_as_uint(__fdiv_rn(fmaxf(amax, 1.0e-10f), 448.0f));
    int const e = min(
        max((int)((b >> 23) & 0xFF) + ((b & 0x7FFFFF) != 0 ? 1 : 0), 1), 254);
    return __uint_as_float((uint32_t)e << 23);
  } else {
    return fmaxf(amax, 1.0e-10f) * (1.0f / 448.0f);
  }
}

// ZOFF: the output's offset in the exchange region
template <size_t ZOFF, int L, bool UE8M0>
__device__ __forceinline__ void group_quant_task(G const &g,
                                                 __nv_bfloat16 const *x,
                                                 uint32_t *done_cnt,
                                                 int kt) {
  __shared__ __align__(16) float scale_s[T];
  // TM: the tokens of the thread mapping (the 8 warps take TM / 8 tokens
  // each): T, or 8 in a layer of fewer tokens, where warp w has token w for
  // w < T and the other warps none
  constexpr int TM = T < 8 ? 8 : T;
  constexpr int TPW = TM / 8; // tokens per warp (all 8 warps)
  static_assert(TM % 8 == 0 && T * 4 / 16 <= 256,
                "group_quant: the 8 warps take T / 8 tokens each");
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  // The previous task finished its scale_s reads before its final barrier.
  // warp w: tokens TPW w .. TPW w + TPW - 1, a lane 4 columns of each; the
  // warp's loads first, all in flight at once (one round trip, not one per
  // token)
  uint2 raw[TPW];
#pragma unroll
  for (int i = 0; i < TPW; i++) { // Bypass L1 because another SM can produce x.
    if (T < 8 && TPW * warp + i >= T) { // no token on this warp
      raw[i] = make_uint2(0u, 0u);
      continue;
    }
    asm volatile("ld.relaxed.gpu.global.v2.u32 {%0, %1}, [%2];"
                 : "=r"(raw[i].x), "=r"(raw[i].y)
                 : "l"(x + (size_t)(TPW * warp + i) * L + kt * 128 + 4 * lane)
                 : "memory");
  }
  // the amax of every token of the warp first: TPW independent shuffle chains,
  // interleaved (one token after the other: a chain of shuffles per token)
  float2 a[TPW], b[TPW];
  float amax[TPW];
#pragma unroll
  for (int i = 0; i < TPW; i++) {
    a[i] = __bfloat1622float2(
        *reinterpret_cast<__nv_bfloat162 const *>(&raw[i].x));
    b[i] = __bfloat1622float2(
        *reinterpret_cast<__nv_bfloat162 const *>(&raw[i].y));
    amax[i] = fmaxf(fmaxf(fabsf(a[i].x), fabsf(a[i].y)),
                    fmaxf(fabsf(b[i].x), fabsf(b[i].y)));
  }
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) { // amax of the 32 lanes
#pragma unroll
    for (int i = 0; i < TPW; i++) {
      amax[i] = fmaxf(amax[i], __shfl_xor_sync(0xffffffffu, amax[i], d));
    }
  }
#pragma unroll
  for (int i = 0; i < TPW; i++) {
    int const t = TPW * warp + i;
    float const s = group_scale<UE8M0>(amax[i]);
    float const r = __frcp_rn(s);
    uint32_t const q = (uint32_t)cvt_e4m3_nv(gq_div(a[i].x, s, r)) |
                       ((uint32_t)cvt_e4m3_nv(gq_div(a[i].y, s, r)) << 8) |
                       ((uint32_t)cvt_e4m3_nv(gq_div(b[i].x, s, r)) << 16) |
                       ((uint32_t)cvt_e4m3_nv(gq_div(b[i].y, s, r)) << 24);
    bool const live = T >= 8 || t < T; // (T < 8: this warp's token)
    if (lane == 0 && live) {
      scale_s[t] = s;
    }
    uint32_t const q1 = __shfl_down_sync(0xffffffffu, q, 1),
                   q2 = __shfl_down_sync(0xffffffffu, q, 2),
                   q3 = __shfl_down_sync(0xffffffffu, q, 3);
    if ((lane & 3) == 0 && live) {
      *reinterpret_cast<uint4 *>(g.rv + ZOFF + (size_t)t * L + kt * 128 +
                                 4 * lane + rg_set()) =
          make_uint4(q, q1, q2, q3);
    }
  }
  __syncthreads();
  if constexpr (T >= 4) {
    if (threadIdx.x < T * 4 / 16) { // the T scales of this K tile: two stores
      *reinterpret_cast<uint4 *>(
          g.rv + ZOFF + (size_t)T * L + (size_t)kt * T * 4 + threadIdx.x * 16 +
          rg_set()) = reinterpret_cast<uint4 const *>(scale_s)[threadIdx.x];
    }
  } else if (threadIdx.x < T) { // T = 1, 2: 4 T bytes, a store per token
    *reinterpret_cast<float *>(g.rv + ZOFF + (size_t)T * L +
                               (size_t)kt * T * 4 + threadIdx.x * 4 +
                               rg_set()) = scale_s[threadIdx.x];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(done_cnt, 1); // Publish all threads' stores after the CTA barrier.
  }
}

// The same for token tile z of NZ (grid y; T >= 16): the tokens [TT z, TT z +
// TT), TT = T / NZ a multiple of 8, the 8 warps TT / 8 tokens each. At T = 64 a
// task of all tokens quantizes 8 tokens a warp on L / 128 SMs while the others
// idle (the DSA layer's v, 32 tasks: 6.9 us after its last input; 4 tiles: 2.4)
template <size_t ZOFF, int L, int NZ, bool UE8M0>
__device__ __forceinline__ void group_quant_tile_task(G const &g,
                                                      __nv_bfloat16 const *x,
                                                      uint32_t *done_cnt,
                                                      int kt,
                                                      int z) {
  __shared__ __align__(16) float scale_s[T];
  constexpr int TT = T / NZ, TPW = TT / 8; // tokens per warp (all 8 warps)
  static_assert(NZ > 1 && NZ * TT == T && TT % 8 == 0,
                "group_quant: token tiles of whole 8-token groups");
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31, t0 = TT * z;
  // The previous task finished its scale_s reads before its final barrier.
  // warp w: tokens TPW w .. TPW w + TPW - 1, a lane 4 columns of each; the
  // warp's loads first, all in flight at once (one round trip, not one per
  // token)
  uint2 raw[TPW];
#pragma unroll
  for (int i = 0; i < TPW; i++) { // Bypass L1 because another SM can produce x.
    asm volatile("ld.relaxed.gpu.global.v2.u32 {%0, %1}, [%2];"
                 : "=r"(raw[i].x), "=r"(raw[i].y)
                 : "l"(x + (size_t)(t0 + TPW * warp + i) * L + kt * 128 +
                       4 * lane)
                 : "memory");
  }
  // the amax of every token of the warp first: TPW independent shuffle chains,
  // interleaved (one token after the other: a chain of shuffles per token)
  float2 a[TPW], b[TPW];
  float amax[TPW];
#pragma unroll
  for (int i = 0; i < TPW; i++) {
    a[i] = __bfloat1622float2(
        *reinterpret_cast<__nv_bfloat162 const *>(&raw[i].x));
    b[i] = __bfloat1622float2(
        *reinterpret_cast<__nv_bfloat162 const *>(&raw[i].y));
    amax[i] = fmaxf(fmaxf(fabsf(a[i].x), fabsf(a[i].y)),
                    fmaxf(fabsf(b[i].x), fabsf(b[i].y)));
  }
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) { // amax of the 32 lanes
#pragma unroll
    for (int i = 0; i < TPW; i++) {
      amax[i] = fmaxf(amax[i], __shfl_xor_sync(0xffffffffu, amax[i], d));
    }
  }
#pragma unroll
  for (int i = 0; i < TPW; i++) {
    int const t = TPW * warp + i; // (of the tile)
    float const s = group_scale<UE8M0>(amax[i]);
    float const r = __frcp_rn(s);
    uint32_t const q = (uint32_t)cvt_e4m3_nv(gq_div(a[i].x, s, r)) |
                       ((uint32_t)cvt_e4m3_nv(gq_div(a[i].y, s, r)) << 8) |
                       ((uint32_t)cvt_e4m3_nv(gq_div(b[i].x, s, r)) << 16) |
                       ((uint32_t)cvt_e4m3_nv(gq_div(b[i].y, s, r)) << 24);
    if (lane == 0) {
      scale_s[t] = s;
    }
    uint32_t const q1 = __shfl_down_sync(0xffffffffu, q, 1),
                   q2 = __shfl_down_sync(0xffffffffu, q, 2),
                   q3 = __shfl_down_sync(0xffffffffu, q, 3);
    if ((lane & 3) == 0) {
      *reinterpret_cast<uint4 *>(g.rv + ZOFF + (size_t)(t0 + t) * L +
                                 kt * 128 + 4 * lane + rg_set()) =
          make_uint4(q, q1, q2, q3);
    }
  }
  __syncthreads();
  if (threadIdx.x < TT * 4 / 16) { // the tile's scales of this K tile
    *reinterpret_cast<uint4 *>(
        g.rv + ZOFF + (size_t)T * L + ((size_t)kt * T + t0) * 4 +
        threadIdx.x * 16 + rg_set()) =
        reinterpret_cast<uint4 const *>(scale_s)[threadIdx.x];
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
      cnt_wait(cnt_now(g) + X::counter, X::x * X::y * X::z);
    }
    __syncthreads();
  }
  constexpr bool UE8M0 = param_or<PARAMS, 1, 0>() != 0;
  if constexpr (SELF::y == 1) {
    group_quant_task<exchange_offset[SELF::buf], L, UE8M0>(
        g,
        buf_at<__nv_bfloat16 const>(g, X::x > 0 ? X::buf : XS),
        cnt_now(g) + SELF::counter,
        tk.x);
  } else {
    group_quant_tile_task<exchange_offset[SELF::buf], L, SELF::y, UE8M0>(
        g,
        buf_at<__nv_bfloat16 const>(g, X::x > 0 ? X::buf : XS),
        cnt_now(g) + SELF::counter,
        tk.x,
        tk.y);
  }
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
