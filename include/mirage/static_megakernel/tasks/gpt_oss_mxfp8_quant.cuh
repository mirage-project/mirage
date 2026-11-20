// tasks/gpt_oss_mxfp8_quant.cuh -- Quantize GPT-OSS h2 for MXFP4 experts.
//
// Input:  fp16/bf16 [T, L] (GPT-OSS-120B: L=2880).
// Output: E4M3 [T, PAD] followed by (PAD / 128) 512-byte scale chunks. Each
//         token occupies 16 bytes in a chunk; the first four bytes hold the
//         UE8M0 scales for its four groups of 32 values and the other 12 bytes
//         are zero. GPT-OSS-120B uses PAD=2944 (23 chunks).
// One task handles 128 columns. It derives one power-of-two scale per 32
// values using the same exponent rule as sum_quant_send.cuh.
#pragma once

#include "../core.cuh"
#include <cuda_fp16.h>
#include <type_traits>

namespace static_mk {

__host__ __device__ constexpr size_t gpt_oss_mxfp8_zq_bytes(int tokens,
                                                            int width) {
  return (size_t)tokens * width + (size_t)(width / 128) * SF_CHUNK;
}

template <size_t ZOFF, int L, int PAD, bool FP16>
__device__ __forceinline__ void gpt_oss_mxfp8_quant_task(G const &g,
                                                         void const *input,
                                                         uint32_t *done_cnt,
                                                         int kt) {
  constexpr int TPW = T / 8;
  static_assert(T == 8 || T == 16 || T == 32,
                "gpt_oss_mxfp8_quant: supported token buckets are 8/16/32");
  static_assert(L == 2880 && PAD == 2944,
                "gpt_oss_mxfp8_quant: GPT-OSS-120B h2 contract");
  using input_t = typename std::conditional<FP16, __half, __nv_bfloat16>::type;
  using input2_t =
      typename std::conditional<FP16, __half2, __nv_bfloat162>::type;
  input_t const *x = reinterpret_cast<input_t const *>(input);
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  size_t const out = ZOFF + rg_set();
  size_t const scale_chunk = out + (size_t)T * PAD + (size_t)kt * SF_CHUNK;

  // Eight warps cover the token rows; each warp loops over T/8 rows. A lane
  // loads and stores four adjacent values, so each eight-lane subgroup owns
  // one MXFP8 (K=32) scale group.
#pragma unroll
  for (int i = 0; i < TPW; ++i) {
    int const t = TPW * warp + i;
    int const col = kt * 128 + lane * 4;
    uint2 raw;
    if (col + 4 <= L) {
      asm volatile("ld.relaxed.gpu.global.v2.u32 {%0, %1}, [%2];"
                   : "=r"(raw.x), "=r"(raw.y)
                   : "l"(x + (size_t)t * L + col)
                   : "memory");
    } else {
      raw = make_uint2(0, 0);
    }

    float2 a, b;
    if constexpr (FP16) {
      a = __half22float2(*reinterpret_cast<input2_t const *>(&raw.x));
      b = __half22float2(*reinterpret_cast<input2_t const *>(&raw.y));
    } else {
      a = __bfloat1622float2(*reinterpret_cast<input2_t const *>(&raw.x));
      b = __bfloat1622float2(*reinterpret_cast<input2_t const *>(&raw.y));
    }
    float const v0 = a.x, v1 = a.y, v2 = b.x, v3 = b.y;
    float amax =
        fmaxf(fmaxf(fabsf(v0), fabsf(v1)), fmaxf(fabsf(v2), fabsf(v3)));
#pragma unroll
    for (int d = 4; d > 0; d >>= 1) {
      amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, d));
    }
    int ex = ceil_log2_bits(fmaxf(amax, 1.0e-30f) * (1.0f / 448.0f));
    ex = ex < -127 ? -127 : (ex > 127 ? 127 : ex);
    float const scale =
        __uint_as_float((uint32_t)(127 - ex) << 23); // 2^-ex, as sum_quant_send
    uint32_t const q = (uint32_t)cvt_e4m3(v0 * scale) |
                       ((uint32_t)cvt_e4m3(v1 * scale) << 8) |
                       ((uint32_t)cvt_e4m3(v2 * scale) << 16) |
                       ((uint32_t)cvt_e4m3(v3 * scale) << 24);
    *reinterpret_cast<uint32_t *>(g.rv + out + (size_t)t * PAD + col) = q;
    if ((lane & 7) == 0) {
      g.rv[scale_chunk + (size_t)t * 16 + (lane >> 3)] = (uint8_t)(ex + 127);
    }
  }

  // The exchange buffer starts each launch as 0xFF. Zero the rest of this
  // scale chunk after the four scale bytes per live token have been written.
  __syncthreads();
  if (threadIdx.x < 32) {
    int const row = threadIdx.x;
    uint32_t *dst = reinterpret_cast<uint32_t *>(g.rv + scale_chunk + row * 16);
    int const first_zero = row < T ? 1 : 0;
    for (int word = first_zero; word < 4; ++word) {
      dst[word] = 0;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(done_cnt, 1);
  }
}

// One task per padded 128-column tile. PARAMS {L, PAD, fp16 flag}. Input is
// h2, either a graph tensor or a producer buffer; output is the exchange
// region's packed MXFP8 activation bank.
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class X,
          class... IN>
__device__ __forceinline__ void run_gpt_oss_mxfp8_quant(Maps const &,
                                                        G const &g,
                                                        KernelLocals &,
                                                        StaticTask const &tk) {
  constexpr int L = PARAMS::v[0], PAD = PARAMS::v[1], FP16 = PARAMS::v[2],
                XS = slot_at<BUF_SLOTS, 0>();
  static_assert(SELF::x * 128 == PAD && L == 2880 && PAD == 2944,
                "gpt_oss_mxfp8_quant: one task per GPT-OSS h2 tile");
  static_assert(SELF::buf >= 0 &&
                    exchange_bytes[SELF::buf] == gpt_oss_mxfp8_zq_bytes(T, PAD),
                "gpt_oss_mxfp8_quant: output must be the packed exchange bank");
  static_assert(SELF::counter >= 0,
                "gpt_oss_mxfp8_quant: output completion counter is required");
  static_assert(
      XS >= 0 || X::buf >= 0,
      "gpt_oss_mxfp8_quant: h2 needs a tensor slot or producer buffer");
  if constexpr (X::x > 0) {
    static_assert(X::counter >= 0,
                  "gpt_oss_mxfp8_quant: producer needs a completion counter");
    if (threadIdx.x == 0) {
      cnt_wait(cnt_now(g) + X::counter, X::x * X::y * X::z);
    }
    __syncthreads();
  }
  gpt_oss_mxfp8_quant_task<exchange_offset[SELF::buf], L, PAD, FP16>(
      g,
      XS >= 0 ? static_cast<void const *>(buf_at<uint8_t const>(g, XS))
              : static_cast<void const *>(buf_at<uint8_t const>(g, X::buf)),
      cnt_now(g) + SELF::counter,
      tk.x);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_gpt_oss_mxfp8_quant() {
  return 0;
}

} // namespace static_mk
