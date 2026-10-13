// tasks/hc_mix.cuh -- the first half of mHC's pre step (hc_pre.cuh is the
// second): task s of grid (KS, 1, 1) takes columns [s KL, s KL + KL) of the
// flattened residual streams R [T][HC H] (bf16; KL = HC H / KS) and, for every
// token, the partial products of those columns with the NMIX rows of fn (fp32
// [NMIX][HC H]) plus the partial sum of squares of R. Warp w = token w, so the
// fn slice is read by the 8 warps of one SM (from L1) and from HBM once.
// Output (the node's buffer, 0xFF-prefilled): [KS][T][HC_MIX_STRIDE] fp32,
// entries 0 .. NMIX - 1 the partial mixes, NMIX the partial sum of squares,
// the rest 0; hc_pre polls them and adds the KS parts in order.
#pragma once
#include "../core.cuh"

namespace static_mk {

constexpr int HC_MIX_STRIDE = 32; // floats per (part, token) in hc_mix's output

template <int KS, int HCH, int NMIX>
__device__ __forceinline__ void
    hc_mix_task(__nv_bfloat16 const *res, float const *fn, float *out, int s) {
  constexpr int KL = HCH / KS;
  static_assert(KL * KS == HCH && KL % 128 == 0,
                "hc_mix: whole 128-column slices (32 lanes x 4 columns)");
  static_assert(NMIX + 1 <= HC_MIX_STRIDE,
                "hc_mix: the mixes and the sum of squares fit one row");
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (warp < T) {
    int const t = warp;
    float acc[NMIX + 1];
#pragma unroll
    for (int r = 0; r <= NMIX; r++) {
      acc[r] = 0.f;
    }
    __nv_bfloat16 const *row = res + (size_t)t * HCH + (size_t)s * KL;
    float const *fs = fn + (size_t)s * KL;
#pragma unroll 1
    for (int c = lane * 4; c < KL; c += 128) {
      uint2 const u = *reinterpret_cast<uint2 const *>(row + c);
      float2 const a =
          __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const *>(&u.x));
      float2 const b =
          __bfloat1622float2(*reinterpret_cast<__nv_bfloat162 const *>(&u.y));
      acc[NMIX] += a.x * a.x + a.y * a.y + b.x * b.x + b.y * b.y;
#pragma unroll
      for (int r = 0; r < NMIX; r++) {
        float4 const w =
            __ldg(reinterpret_cast<float4 const *>(fs + (size_t)r * HCH + c));
        acc[r] += w.x * a.x + w.y * a.y + w.z * b.x + w.w * b.y;
      }
    }
#pragma unroll
    for (int r = 0; r <= NMIX; r++) {
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) {
        acc[r] += __shfl_xor_sync(0xffffffffu, acc[r], o);
      }
    }
    // lane j writes entry j (one 4-B store each: hc_pre polls them)
    float v = 0.f;
#pragma unroll
    for (int r = 0; r <= NMIX; r++) {
      if (lane == r) {
        v = acc[r];
      }
    }
    out[((size_t)s * T + t) * HC_MIX_STRIDE + lane] = v;
  }
  __syncthreads();
}

// hc_mix task x of SELF::x: PARAMS {HC, H, NMIX}; BUF_SLOTS {the residual
// streams [T][HC H] bf16, fn [NMIX][HC H] fp32 (graph tensors)}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void
    run_hc_mix(Maps const &, G const &g, KernelLocals &, StaticTask const &tk) {
  constexpr int HC = PARAMS::v[0], H = PARAMS::v[1], NMIX = PARAMS::v[2];
  static_assert(NMIX == HC * (HC + 2),
                "hc_mix: params {HC, H, NMIX = HC (HC + 2)}");
  static_assert(SELF::buf >= 0, "hc_mix: its output buffer");
  hc_mix_task<SELF::x, HC * H, NMIX>(
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<float>(g, SELF::buf),
      tk.x);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_hc_mix() {
  return 0;
}

} // namespace static_mk
