// tasks/hc_mix.cuh -- the projection of DeepSeek-V4.1's mHC coefficients
// (model.py Block.hc_mixes), run beside the sublayer's body: its pre, post and
// comb come from the streams the sublayer starts from, and nothing before the
// sublayer's end needs them (its input was collapsed with the previous
// sublayer's pre). Task s of grid (H / 32, 1, 1) = columns [32 s, 32 s + 32) of
// every stream of every token: the partial mixes [T][NMIX] = R [T][HC 32] x
// fn [HC 32][NMIX], as step 4 of hc_post_mix.cuh (a warp per 8 rows of the
// mixes, fn as bf16 hi + lo), entry NMIX the sum of squares of the token's HC
// 32 values, then 0s, into row (s, t) of the node's buffer
// ([KS][T][HC_MIX_STRIDE] fp32, 0xFF before the launch, polled by hc_coef).
#pragma once
#include "../core.cuh"
#include "hc_post_mix.cuh"

namespace static_mk {

template <int H, int HC, int NMIX>
__device__ __forceinline__ void hc_mix_task(__nv_bfloat16 const *res,
                                            float const *fn,
                                            float *mix,
                                            int s,
                                            char *scratch) {
  constexpr int C = 32;
  static_assert(T == 8 && H % C == 0 && C == HC_MIX_STRIDE &&
                    NMIX + 1 <= HC_MIX_STRIDE && NMIX % 8 == 0 &&
                    (NMIX * HC * C) % 256 == 0 &&
                    (NMIX / 8) <= 8 && HC_MIX_STRIDE - NMIX <= 256 / T,
                "hc_mix: 8 tokens, whole 8-row tiles of the mixes");
  constexpr int KP = hcpm_kp<HC>(), FNV = NMIX * HC * C / 256;
  // scratch: R [T][KP] bf16, fn's slice as bf16 hi and lo [NMIX][KP] (row k =
  // stream j, column c: k = 32 j + c)
  __nv_bfloat16 *rb = reinterpret_cast<__nv_bfloat16 *>(scratch);
  __nv_bfloat16 *fh = rb + T * KP, *fl = fh + NMIX * KP;
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int const c = s * C + lane;
  __syncthreads(); // the ring area is free once every warp has left the
                   // previous task
  // loops, not unrolled: a kernel's code is fetched cold at each launch, and
  // these tasks have time (they run beside the sublayer)
#pragma unroll 1
  for (int q = 0; q < FNV; q++) {
    int const idx = threadIdx.x + 256 * q, r = idx / (HC * C),
              k = idx % (HC * C);
    float const f =
        __ldg(fn + (size_t)r * HC * H + (size_t)(k / C) * H + s * C + k % C);
    __nv_bfloat16 const hi = __float2bfloat16_rn(f);
    fh[r * KP + k] = hi;
    fl[r * KP + k] = __float2bfloat16_rn(f - __bfloat162float(hi));
  }
#pragma unroll 1
  for (int i = 0; i < HC; i++) { // token `warp`, this lane's column
    rb[warp * KP + i * C + lane] = res[((size_t)warp * HC + i) * H + c];
  }
  __syncthreads();
  constexpr int NT8 = NMIX / 8;
  if (warp < NT8) { // tokens 0 .. 7 x rows [8 warp, 8 warp + 8)
    int const n0 = warp * 8, g8 = lane >> 2, tq = lane & 3;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 1
    for (int k0 = 0; k0 < HC * C; k0 += 16) {
      uint32_t const a[4] = {hcpm_ld2(rb + g8 * KP + k0 + 2 * tq),
                             hcpm_ld2(rb + g8 * KP + k0 + 2 * tq),
                             hcpm_ld2(rb + g8 * KP + k0 + 8 + 2 * tq),
                             hcpm_ld2(rb + g8 * KP + k0 + 8 + 2 * tq)};
      uint32_t const bh[2] = {hcpm_ld2(fh + (n0 + g8) * KP + k0 + 2 * tq),
                              hcpm_ld2(fh + (n0 + g8) * KP + k0 + 8 + 2 * tq)};
      uint32_t const bl[2] = {hcpm_ld2(fl + (n0 + g8) * KP + k0 + 2 * tq),
                              hcpm_ld2(fl + (n0 + g8) * KP + k0 + 8 + 2 * tq)};
      hcpm_mma(a, bh, acc);
      hcpm_mma(a, bl, acc);
    }
    *reinterpret_cast<float2 *>(mix + ((size_t)s * T + g8) * HC_MIX_STRIDE +
                                n0 + 2 * tq) = make_float2(acc[0], acc[1]);
  }
  // entry NMIX: the sum of squares of the token's HC 32 values, then 0s
  constexpr int TPT = 256 / T, PER = HC * C / TPT;
  int const t = threadIdx.x / TPT, part = threadIdx.x % TPT;
  float q = 0.f;
#pragma unroll 1
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
  __syncthreads();
}

// hc_mix task x. PARAMS {HC, H, NMIX}; BUF_SLOTS {the streams R ([T][HC H]
// bf16), fn [NMIX][HC H] fp32}; its buffer: the mixes
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_hc_mix(Maps const &,
                                           G const &g,
                                           KernelLocals &L,
                                           StaticTask const &tk) {
  constexpr int HC = PARAMS::v[0], H = PARAMS::v[1], NMIX = PARAMS::v[2];
  static_assert(NMIX == HC * (HC + 2),
                "hc_mix: params {HC, H, NMIX = HC (HC + 2)}");
  static_assert(SELF::x * 32 == H && SELF::buf >= 0,
                "hc_mix: a task per 32 columns, its mixes");
  hc_mix_task<H, HC, NMIX>(buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
                           buf_at<float const>(g, slot_at<BUF_SLOTS, 1>()),
                           buf_at<float>(g, SELF::buf),
                           tk.x,
                           L.sm);
}

// dynamic shared memory: R and fn's slice, inside the ring area
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_hc_mix() {
  static_assert((T + 2 * PARAMS::v[2]) * hcpm_kp<PARAMS::v[0]>() * 2 <=
                    RING_BYTES,
                "hc_mix: its scratch in the ring area");
  return RING_BYTES;
}

} // namespace static_mk
