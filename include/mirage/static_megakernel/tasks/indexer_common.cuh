// tasks/indexer_common.cuh -- what DeepSeek-V4.1's indexer tasks share
// (indexer_q, index_k_post, indexer_score, indexer_topk): fp4_act_quant with
// E8M0 scales (kernel.py fp4_quant_kernel, inplace: the dequantized values
// kept as bf16), the index K cache row, and the e2m1 -> bf16 conversion.
#pragma once
#include "../core.cuh"

namespace static_mk {

// the index K cache [T][CAP][IX_ROW] uint8: per compressed position 128 e2m1
// values (64 bytes, element 2i in the low nibble), then 4 e8m0 scale bytes
// (byte j: columns 32 j .. 32 j + 31)
constexpr int IX_HD = 128, IX_ROW = IX_HD / 2 + IX_HD / 32;

// fp4_quant_kernel's scale of a group of 32: amax floored at 6 * 2^-126, times
// fp32(1 / 6), ceil(log2) from the bits (fast_round_scale) -> e, the scale
// 2^e
__device__ __forceinline__ int fp4_exponent(float amax) {
  float const a = __fmul_rn(fmaxf(amax, 6.0f * 0x1p-126f), 1.0f / 6.0f);
  uint32_t const b = __float_as_uint(a);
  return (int)((b >> 23) & 0xFFu) - 127 + ((b & 0x7FFFFFu) != 0u ? 1 : 0);
}
__device__ __forceinline__ float pow2f(int e) { // 2^e, -126 <= e <= 127
  return __uint_as_float((uint32_t)(e + 127) << 23);
}

// 4 values (bf16 values as floats) of a group with scale 2^e -> 2 bytes of
// e2m1 (x[0] in byte 0's low nibble): x / 2^e (exact: a power of 2), clamped
// to +-6 (satfinite) and rounded to nearest even, as the official's
// T.Cast(FP4, clamp(x / s)); deq: the e2m1 values times 2^e (exact), what the
// official keeps (inplace)
__device__ __forceinline__ uint32_t fp4_quant4(float const *x,
                                               int e,
                                               float *deq) {
  float const inv = pow2f(-e), s = pow2f(e);
  uint32_t h0, h1;
  uint16_t r;
  asm("{\n.reg .b8 b0, b1;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b0, %4, %3;\n"
      "cvt.rn.satfinite.e2m1x2.f32 b1, %6, %5;\n"
      "mov.b16 %0, {b0, b1};\n"
      "cvt.rn.f16x2.e2m1x2 %1, b0;\n"
      "cvt.rn.f16x2.e2m1x2 %2, b1;\n}"
      : "=h"(r), "=r"(h0), "=r"(h1)
      : "f"(x[0] * inv), "f"(x[1] * inv), "f"(x[2] * inv), "f"(x[3] * inv));
  float2 const f0 = __half22float2(*reinterpret_cast<__half2 const *>(&h0)),
               f1 = __half22float2(*reinterpret_cast<__half2 const *>(&h1));
  deq[0] = f0.x * s;
  deq[1] = f0.y * s;
  deq[2] = f1.x * s;
  deq[3] = f1.y * s;
  return r;
}

// one byte of e2m1 x2 (low nibble first) times s (a power of 2) -> bf16 x2
// (exact: at most 2 significant bits)
__device__ __forceinline__ uint32_t fp4_bf16x2(uint32_t byte, float s) {
  uint32_t h;
  asm("{\n.reg .b8 b;\ncvt.u8.u32 b, %1;\ncvt.rn.f16x2.e2m1x2 %0, b;\n}"
      : "=r"(h)
      : "r"(byte));
  float2 const f = __half22float2(*reinterpret_cast<__half2 const *>(&h));
  __nv_bfloat162 const v = __floats2bfloat162_rn(f.x * s, f.y * s);
  return *reinterpret_cast<uint32_t const *>(&v);
}

} // namespace static_mk
