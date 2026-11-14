// tasks/sm90_mxfp4.cuh -- the Hopper MXFP4 expert weights of the static
// megakernel's MoE (sm90_moe_pool): their layout and the helpers that decode
// them into mma.sync A fragments.
//
// The weights come repacked (static_megakernel.py sm90_moe_pack) as blocks
// [E][N / ROWS][K tiles of 128][ROWS x 68 B], contiguous in the order a unit
// streams them: a block is its rows' packed values (ROWS x 64 B, the bytes of a
// row permuted so that MMA thread q's 16 (the A fragment's K columns 16 kb + 2q
// (+ 8), kb = 0..7) are contiguous) then their E8M0 scales (ROWS x 4 B). W13's
// blocks are 128 rows: the 64 gate rows and the 64 up rows of the same 64
// outputs.
#pragma once
#include "sm90_common.cuh"
#include "sm90_frag.cuh"

namespace static_mk {

#ifdef STATIC_SM90
// The f32 factor 2^(scale - 121) of an E8M0 scale (any up to 248): it takes
// the fp32 sum of the scale's 32-value block of E4M3 values v 2^-6
// (moe_decode_word) to the sum of v 2^(scale - 127) (exact: a power of two). A
// zero scale gives 2^-121 for 0 (the K padding's weights are 0).
__device__ __forceinline__ float moe_scale_f32(uint32_t scale) {
  return __uint_as_float((scale + 6u) << 23);
}
// Eight packed E2M1 values to four f16 pairs (unscaled: moe_scale_f32). The
// packing (sm90_moe_pack) puts register c's two values in the low nibbles of
// bytes 2c, 2c + 1 (c = 0, 1) or the high nibbles of bytes 2c - 4, 2c - 3
// (c = 2, 3); a nibble s.ee.m as the E4M3 byte s.0ee.m00 is its value times
// 2^-6 (exact, the zero exponent included), so two shift-and-masks give eight
// E4M3 bytes in register order and the hardware converts them to f16 pairs.
__device__ __forceinline__ void moe_decode_word(uint32_t w,
                                                uint32_t (&out)[4]) {
  uint32_t const lo = ((w << 2) & 0x1c1c1c1cu) | ((w << 4) & 0x80808080u);
  uint32_t const hi = ((w >> 2) & 0x1c1c1c1cu) | (w & 0x80808080u);
  asm("{\n.reg .b16 a, b, c, d;\n"
      "mov.b32 {a, b}, %4;\n"
      "mov.b32 {c, d}, %5;\n"
      "cvt.rn.f16x2.e4m3x2 %0, a;\n"
      "cvt.rn.f16x2.e4m3x2 %1, b;\n"
      "cvt.rn.f16x2.e4m3x2 %2, c;\n"
      "cvt.rn.f16x2.e4m3x2 %3, d;\n}"
      : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
      : "r"(lo), "r"(hi));
}
// d (16 weight rows x 8 tokens, fp32) += A (16 x K16) * B (K16 x 8 tokens),
// f16
__device__ __forceinline__ void mma_m16n8k16(float (&d)[4],
                                             uint32_t const (&a)[4],
                                             uint32_t b0,
                                             uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
               "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// a contiguous copy global -> shared, completing on mbarrier mb
__device__ __forceinline__ void
    bulk_g2s(void const *src, uint32_t mb, uint32_t sm, uint32_t bytes) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::"
               "bytes [%0], [%1], %2, [%3];" ::"r"(sm),
               "l"((uint64_t)src),
               "r"(bytes),
               "r"(mb)
               : "memory");
}
#endif // STATIC_SM90

} // namespace static_mk
