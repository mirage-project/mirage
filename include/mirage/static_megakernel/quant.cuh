// quant.cuh -- the quantization of latent columns [128 kt, 128 kt + 128): add latent_down's NSPLIT partial sums (zpart, in a fixed
// order so that every GPU gets the same z), MXFP8 per (token, 32 columns): z_q e4m3 bytes + one e8m0 scale byte, then send the
// tile's bytes and its scale chunk to every GPU (multicast; the receivers poll the bytes, 0xFF-prefilled). Warps 0..3 work.
#pragma once
#include "runtime.cuh"
#include "moe_types.cuh"

namespace static_mk {

template <int NSPLIT>
__device__ __forceinline__ void quant_task(G const &g, int kt) {
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __syncthreads();
  if (warp < 4) {   // warp w: tokens 2w and 2w + 1; lane: columns 4 lane .. 4 lane + 3 of the tile (8 lanes = one 32-column group)
    float4 zp[2][NSPLIT];
    bool all;
    do {   // the NSPLIT partial sums of both tokens, loaded together; re-load all until every part has landed
#pragma unroll
      for (int i = 0; i < 2; i++)
#pragma unroll
        for (int p_ = 0; p_ < NSPLIT; p_++) zp[i][p_] = ldf4_relaxed(g.zpart + ((size_t)p_ * T + 2 * warp + i) * LAT + kt * 128 + 4 * lane);
      all = true;
#pragma unroll
      for (int i = 0; i < 2; i++)
#pragma unroll
        for (int p_ = 0; p_ < NSPLIT; p_++) all = all && validf4(zp[i][p_]);
    } while (!all);
#pragma unroll
    for (int i = 0; i < 2; i++) {
      int const t = 2 * warp + i;
      float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int p_ = 0; p_ < NSPLIT; p_++) { z.x += zp[i][p_].x; z.y += zp[i][p_].y; z.z += zp[i][p_].z; z.w += zp[i][p_].w; }
      float amax = fmaxf(fmaxf(fabsf(z.x), fabsf(z.y)), fmaxf(fabsf(z.z), fabsf(z.w)));
      for (int d = 4; d > 0; d >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));   // over the 8 lanes of the group
      amax = fmaxf(amax, 1.0e-30f);
      float e = ceilf(log2f(amax * (1.0f / 448.0f)));   // scale exponent: amax / 2^e <= 448 (e4m3 max)
      e = fminf(fmaxf(e, -127.f), 127.f);
      float const sc = exp2f(-e);
      uint32_t const q = (uint32_t)cvt_e4m3(z.x * sc) | ((uint32_t)cvt_e4m3(z.y * sc) << 8) | ((uint32_t)cvt_e4m3(z.z * sc) << 16) |
                         ((uint32_t)cvt_e4m3(z.w * sc) << 24);
      *reinterpret_cast<uint32_t *>(g.zq + t * LAT + kt * 128 + 4 * lane) = q;
      if ((lane & 7) == 0) g.xsf[kt * SF_CHUNK + t * 16 + (lane >> 3)] = (uint8_t)(int)(e + 127.f);
    }
    asm volatile("bar.sync 1, 128;" ::: "memory");
    {  // the rest of the 512-B scale chunk: zero (no 0xFF byte may stay)
      int const row = threadIdx.x >> 2, part = threadIdx.x & 3;
      if (!(row < T && part == 0)) *reinterpret_cast<uint32_t *>(g.xsf + kt * SF_CHUNK + row * 16 + part * 4) = 0u;
    }
    asm volatile("bar.sync 1, 128;" ::: "memory");
    // to every GPU: the tile's z_q bytes (8 tokens x 128 B) and its scale chunk (512 B)
    if (threadIdx.x < 64) {
      int const t = threadIdx.x >> 3, sg = threadIdx.x & 7;
      size_t const off = RG_ZQ + (size_t)t * LAT + kt * 128 + sg * 16;
      push16(g, off, *reinterpret_cast<uint4 const *>(g.rv + off));
    } else if (threadIdx.x < 96) {
      size_t const off = RG_ZSF + (size_t)kt * SF_CHUNK + (threadIdx.x - 64) * 16;
      push16(g, off, *reinterpret_cast<uint4 const *>(g.rv + off));
    }
  }
  __syncthreads();
}

}  // namespace static_mk
