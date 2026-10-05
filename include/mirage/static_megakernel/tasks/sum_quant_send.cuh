// tasks/sum_quant_send.cuh -- the quantization of latent columns [128 kt, 128
// kt + 128): add latent_down's NSPLIT partial sums (zpart = the latent_down
// node's output, K parts in slots (gemm_tile.cuh COMBINE_SLOTS), in a fixed
// order so that every GPU gets the same z), MXFP8 per (token, 32 columns): z_q
// e4m3 bytes + one e8m0 scale byte, then send the tile's bytes and its scale
// chunk to every GPU (multicast; the receivers poll the bytes, 0xFF-prefilled).
// Warps 0..3 work. The output (the node's buffer, in the exchange region): z_q
// [T][L] e4m3, then the scale chunks [L / 128][512] (zq_scales_at). L = the
// node's params[0] (z's width); its readers find it in their producer's params.
#pragma once
#include "../core.cuh"

namespace static_mk {

// the output's layout for z of width L: z_q bytes, then the scale chunks
__host__ __device__ constexpr size_t zq_scales_at(int L) {
  return (size_t)T * L;
}
__host__ __device__ constexpr size_t zq_bytes(int L) {
  return zq_scales_at(L) + (size_t)(L / 128) * SF_CHUNK;
}

template <int NSPLIT, size_t ZOFF, int LAT> // ZOFF: the output's offset in the
                                            // exchange region; LAT: z's width
                                            __device__ __forceinline__ void
    sum_quant_send_task(G const &g, float *zpart, int kt) {
  constexpr size_t ZQ_SF = zq_scales_at(LAT);
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __syncthreads();
  if (warp < 4) { // warp w: tokens 2w and 2w + 1; lane: columns 4 lane .. 4
                  // lane + 3 of the tile (8 lanes = one 32-column group)
    float4 zp[2][NSPLIT];
    bool all;
    do { // the NSPLIT partial sums of both tokens, loaded together; re-load all
         // until every part has landed
#pragma unroll
      for (int i = 0; i < 2; i++)
#pragma unroll
        for (int p_ = 0; p_ < NSPLIT; p_++) {
          zp[i][p_] =
              ldf4_relaxed(zpart + ((size_t)p_ * T + 2 * warp + i) * LAT +
                           kt * 128 + 4 * lane);
        }
      all = true;
#pragma unroll
      for (int i = 0; i < 2; i++)
#pragma unroll
        for (int p_ = 0; p_ < NSPLIT; p_++) {
          all = all && validf4(zp[i][p_]);
        }
    } while (!all);
#pragma unroll
    for (int i = 0; i < 2; i++) {
      int const t = 2 * warp + i;
      float4 z = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int p_ = 0; p_ < NSPLIT; p_++) {
        z.x += zp[i][p_].x;
        z.y += zp[i][p_].y;
        z.z += zp[i][p_].z;
        z.w += zp[i][p_].w;
      }
      float amax =
          fmaxf(fmaxf(fabsf(z.x), fabsf(z.y)), fmaxf(fabsf(z.z), fabsf(z.w)));
      for (int d = 4; d > 0; d >>= 1) {
        amax =
            fmaxf(amax,
                  __shfl_xor_sync(
                      0xffffffffu, amax, d)); // over the 8 lanes of the group
      }
      amax = fmaxf(amax, 1.0e-30f);
      float e = ceilf(log2f(
          amax *
          (1.0f / 448.0f))); // scale exponent: amax / 2^e <= 448 (e4m3 max)
      e = fminf(fmaxf(e, -127.f), 127.f);
      float const sc = exp2f(-e);
      uint32_t const q = (uint32_t)cvt_e4m3(z.x * sc) |
                         ((uint32_t)cvt_e4m3(z.y * sc) << 8) |
                         ((uint32_t)cvt_e4m3(z.z * sc) << 16) |
                         ((uint32_t)cvt_e4m3(z.w * sc) << 24);
      *reinterpret_cast<uint32_t *>(g.rv + ZOFF + t * LAT + kt * 128 +
                                    4 * lane + rg_set()) = q;
      if ((lane & 7) == 0) {
        g.rv[ZOFF + ZQ_SF + kt * SF_CHUNK + t * 16 + (lane >> 3) + rg_set()] =
            (uint8_t)(int)(e + 127.f);
      }
    }
    asm volatile("bar.sync 1, 128;" ::: "memory");
    { // the rest of the 512-B scale chunk: zero (no 0xFF byte may stay)
      int const row = threadIdx.x >> 2, part = threadIdx.x & 3;
      if (!(row < T && part == 0)) {
        *reinterpret_cast<uint32_t *>(g.rv + ZOFF + ZQ_SF + kt * SF_CHUNK +
                                      row * 16 + part * 4 + rg_set()) = 0u;
      }
    }
    asm volatile("bar.sync 1, 128;" ::: "memory");
    // to every GPU: the tile's z_q bytes (T tokens x 128 B) and its scale chunk
    // (512 B)
    if (threadIdx.x < 64) {
      int const t = threadIdx.x >> 3, sg = threadIdx.x & 7;
      size_t const off = ZOFF + (size_t)t * LAT + kt * 128 + sg * 16;
      push16(g, off, *reinterpret_cast<uint4 const *>(g.rv + off + rg_set()));
    } else if (threadIdx.x < 96) {
      size_t const off =
          ZOFF + ZQ_SF + (size_t)kt * SF_CHUNK + (threadIdx.x - 64) * 16;
      push16(g, off, *reinterpret_cast<uint4 const *>(g.rv + off + rg_set()));
    }
    if constexpr (REARM) { // the one reader of these parts; after z_q is sent
                           // (off the path to the moe_experts tasks)
#pragma unroll
      for (int i = 0; i < 2; i++)
#pragma unroll
        for (int p_ = 0; p_ < NSPLIT; p_++) {
          rearm16(zpart + ((size_t)p_ * T + 2 * warp + i) * LAT + kt * 128 +
                  4 * lane);
        }
    }
  }
  __syncthreads();
}

// sum_quant_send task x = z columns [128 x, 128 x + 128); adds latent_down's
// Z::y partial sums (K parts in slots). PARAMS {L (z's width)}
template <class SELF, class PARAMS, class SLOTS, class Z, class... REST>
__device__ __forceinline__ void run_sum_quant_send(Maps const &,
                                                   G const &g,
                                                   KernelLocals &,
                                                   StaticTask const &tk) {
  constexpr int L = PARAMS::v[0];
  static_assert(
      L % 128 == 0 && Z::v[0] == COMBINE_SLOTS && Z::v[2] == L,
      "sum_quant_send: latent_down's L outputs, its K parts in slots");
  static_assert(SELF::buf >= 0 && exchange_bytes[SELF::buf] == zq_bytes(L),
                "sum_quant_send: its output, z_q, in the exchange region");
  sum_quant_send_task<Z::y, exchange_offset[SELF::buf], L>(
      g, buf_at<float>(g, Z::buf), tk.x);
}

template <class SELF, class PARAMS, class SLOTS, class... IN>
constexpr int smem_sum_quant_send() {
  return 0;
}

} // namespace static_mk
