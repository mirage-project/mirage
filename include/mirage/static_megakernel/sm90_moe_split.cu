// GPT-OSS's routed experts as two kernels after the layer's static megakernel
// (which then ends at the gate; python/mirage/mpk/sm90_moe_split.py): W13 with
// the clamped SwiGLU, then W2 with the top-k combine and the residual.
//
// triton_kernels' matmul_ogs scheme (SGLang's GPT-OSS MoE on Hopper): the
// MXFP4 weights are wgmma's A operand, decoded in registers with the E8M0
// scale folded in, register-A async wgmma (two A buffers in turn: one decoded
// while the other's wgmmas run); B = the expert's routed tokens' activations
// (compact, 8-token groups), each row TMA'd into the 128-B swizzled K tile
// next to the weight tile, in a 4-stage ring a producer warp fills. A CTA is
// one item, (64 outputs, expert): several CTAs an SM, the grid's experts with
// no routed token exit at once. In the layer kernel's pool (sm90_moe_pool)
// ptxas serializes every wgmma (one function at the register cap), hence the
// separate kernels. GPT-OSS-20B at 32 requests (31 experts), H100: W13 + W2 +
// the combine 173 us, triton_kernels' W13 + W2 194 us.
//
// Compile: -DTT= (token rows) -DHH= -DII= -DEE= -DTOPKK=
#include <cstdint>
#include <cstdio>
#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace mk {

constexpr int T = TT, H = HH, I = II, E = EE, TOPK = TOPKK;
constexpr float SUM_SCALE = 4.0f; // the decode's values are 1/4 of the weights'

__device__ __forceinline__ uint32_t su32(void const *p) {
  return (uint32_t)__cvta_generic_to_shared(p);
}
__device__ __forceinline__ void mbar_init(uint32_t a, int n) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" ::"r"(a), "r"(n));
}
__device__ __forceinline__ void mbar_arrive(uint32_t a) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" ::"r"(a) : "memory");
}
__device__ __forceinline__ void mbar_expect(uint32_t a, uint32_t tx) {
  asm volatile(
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(a), "r"(tx)
      : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t a, int ph) {
  asm volatile("{\n.reg .pred P;\nW:\nmbarrier.try_wait.parity.shared::cta.b64 "
               "P, [%0], %1;\n@P bra D;\nbra W;\nD:\n}" ::"r"(a),
               "r"(ph)
               : "memory");
}
__device__ __forceinline__ void
    bulk_g2s(void const *src, uint32_t mb, uint32_t sm, uint32_t bytes) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::"
               "bytes [%0], [%1], %2, [%3];" ::"r"(sm),
               "l"((uint64_t)src),
               "r"(bytes),
               "r"(mb)
               : "memory");
}
__device__ __forceinline__ void
    tma3(void const *d, uint32_t mb, uint32_t sm, int c0, int c1, int c2) {
  asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::"
               "complete_tx::bytes"
               " [%0], [%1, {%3, %4, %5}], [%2];" ::"r"(sm),
               "l"((uint64_t)d),
               "r"(mb),
               "r"(c0),
               "r"(c1),
               "r"(c2)
               : "memory");
}
__device__ __forceinline__ uint64_t denc(uint64_t x) {
  return (x & 0x3FFFFULL) >> 4ULL;
}
// a 128-B swizzled K-major tile, 8-row groups 1024 B apart
__device__ __forceinline__ uint64_t gmma_desc(uint32_t a) {
  return denc(a) | (1ULL << 16) | (denc(1024) << 32) | (1ULL << 62);
}
__device__ __forceinline__ void wgmma_fence() {
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
}
__device__ __forceinline__ void wgmma_commit() {
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
}
template <int N>
__device__ __forceinline__ void wgmma_wait() {
  asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N) : "memory");
}
template <int V>
struct IC {
  static constexpr int value = V;
};

// Eight packed values of one 32-bit word (sm90_moe_pack's order, each byte's
// bits permuted by the host: low-slot nibble at bits {5: sign, 2, 1: exp,
// 0: mantissa}, high-slot at {7, 4, 3, 6}) to four f16 pairs v 2^(s - 129):
// their E4M3 bytes s.0ee.m00 (= v 2^-6) by a shift and a mask each, the
// hardware's E4M3 -> f16, times scale i's f16 2^(s - 123) (sc4: the row's four
// scales' (s - 108) << 2 a byte, every scale in [108, 138]).
__device__ __forceinline__ void
    decode_word(uint32_t w, uint32_t sc4, int i, uint32_t (&out)[4]) {
  uint32_t const lo = (w << 2) & 0x9C9C9C9Cu;
  uint32_t const hi = (w & 0x98989898u) | ((w >> 4) & 0x04040404u);
  uint32_t scale2;
  asm("prmt.b32 %0, %1, 0, %2;"
      : "=r"(scale2)
      : "r"(sc4), "r"(0x0404u | (i << 4) | (i << 12)));
  uint32_t f[4];
  asm("{\n.reg .b16 a, b, c, d;\n"
      "mov.b32 {a, b}, %4;\n"
      "mov.b32 {c, d}, %5;\n"
      "cvt.rn.f16x2.e4m3x2 %0, a;\n"
      "cvt.rn.f16x2.e4m3x2 %1, b;\n"
      "cvt.rn.f16x2.e4m3x2 %2, c;\n"
      "cvt.rn.f16x2.e4m3x2 %3, d;\n}"
      : "=r"(f[0]), "=r"(f[1]), "=r"(f[2]), "=r"(f[3])
      : "r"(lo), "r"(hi));
#pragma unroll
  for (int c = 0; c < 4; c++) {
    __half2 const r = __hmul2(*reinterpret_cast<__half2 const *>(&f[c]),
                              *reinterpret_cast<__half2 const *>(&scale2));
    out[c] = *reinterpret_cast<uint32_t const *>(&r);
  }
}

// d (64 weight rows x N tokens, fp32) (+)= A (registers: a K tile of 128 as 8
// K16 fragments) B (N tokens x K 128, shared, two 64-wide halves b0, b1)
#define WG_A(i) "r"(a[i][0]), "r"(a[i][1]), "r"(a[i][2]), "r"(a[i][3])
template <int N>
__device__ __forceinline__ void wgmma_k128(
    float *d, uint32_t const (&a)[8][4], uint64_t b0, uint64_t b1, int acc);
template <>
__device__ __forceinline__ void wgmma_k128<8>(
    float *d, uint32_t const (&a)[8][4], uint64_t b0, uint64_t b1, int acc) {
  asm volatile("{\n.reg .pred p;\n.reg .b64 d<8>;\nsetp.ne.b32 p, %38, 0;\n"
               "add.s64 d0, %36, 0;\nadd.s64 d1, %36, 2;\nadd.s64 d2, %36, "
               "4;\nadd.s64 d3, %36, 6;\n"
               "add.s64 d4, %37, 0;\nadd.s64 d5, %37, 2;\nadd.s64 d6, %37, "
               "4;\nadd.s64 d7, %37, 6;\n"
               "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, "
               "%3}, {%4, %5, %6, %7}, d0, p, 1, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, "
               "%3}, {%8, %9, %10, %11}, d1, 1, 1, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, "
               "%3}, {%12, %13, %14, %15}, d2, 1, 1, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, "
               "%3}, {%16, %17, %18, %19}, d3, 1, 1, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, "
               "%3}, {%20, %21, %22, %23}, d4, 1, 1, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, "
               "%3}, {%24, %25, %26, %27}, d5, 1, 1, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, "
               "%3}, {%28, %29, %30, %31}, d6, 1, 1, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, "
               "%3}, {%32, %33, %34, %35}, d7, 1, 1, 1, 0;\n}"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : WG_A(0),
                 WG_A(1),
                 WG_A(2),
                 WG_A(3),
                 WG_A(4),
                 WG_A(5),
                 WG_A(6),
                 WG_A(7),
                 "l"(b0),
                 "l"(b1),
                 "r"(acc));
}
template <>
__device__ __forceinline__ void wgmma_k128<16>(
    float *d, uint32_t const (&a)[8][4], uint64_t b0, uint64_t b1, int acc) {
  asm volatile(
      "{\n.reg .pred p;\n.reg .b64 d<8>;\nsetp.ne.b32 p, %42, 0;\n"
      "add.s64 d0, %40, 0;\nadd.s64 d1, %40, 2;\nadd.s64 d2, %40, 4;\nadd.s64 "
      "d3, %40, 6;\n"
      "add.s64 d4, %41, 0;\nadd.s64 d5, %41, 2;\nadd.s64 d6, %41, 4;\nadd.s64 "
      "d7, %41, 6;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, "
      "%5, %6, %7}, {%8, %9, %10, %11}, d0, p, 1, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, "
      "%5, %6, %7}, {%12, %13, %14, %15}, d1, 1, 1, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, "
      "%5, %6, %7}, {%16, %17, %18, %19}, d2, 1, 1, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, "
      "%5, %6, %7}, {%20, %21, %22, %23}, d3, 1, 1, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, "
      "%5, %6, %7}, {%24, %25, %26, %27}, d4, 1, 1, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, "
      "%5, %6, %7}, {%28, %29, %30, %31}, d5, 1, 1, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, "
      "%5, %6, %7}, {%32, %33, %34, %35}, d6, 1, 1, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, "
      "%5, %6, %7}, {%36, %37, %38, %39}, d7, 1, 1, 1, 0;\n}"
      : "+f"(d[0]),
        "+f"(d[1]),
        "+f"(d[2]),
        "+f"(d[3]),
        "+f"(d[4]),
        "+f"(d[5]),
        "+f"(d[6]),
        "+f"(d[7])
      : WG_A(0),
        WG_A(1),
        WG_A(2),
        WG_A(3),
        WG_A(4),
        WG_A(5),
        WG_A(6),
        WG_A(7),
        "l"(b0),
        "l"(b1),
        "r"(acc));
}
template <>
__device__ __forceinline__ void wgmma_k128<24>(
    float *d, uint32_t const (&a)[8][4], uint64_t b0, uint64_t b1, int acc) {
  asm volatile(
      "{\n.reg .pred p;\n.reg .b64 d<8>;\nsetp.ne.b32 p, %46, 0;\n"
      "add.s64 d0, %44, 0;\nadd.s64 d1, %44, 2;\nadd.s64 d2, %44, 4;\nadd.s64 "
      "d3, %44, 6;\n"
      "add.s64 d4, %45, 0;\nadd.s64 d5, %45, 2;\nadd.s64 d6, %45, 4;\nadd.s64 "
      "d7, %45, 6;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16 {%0, %1, %2, %3, %4, "
      "%5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, d0, p, 1, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16 {%0, %1, %2, %3, %4, "
      "%5, %6, %7, %8, %9, %10, %11}, {%16, %17, %18, %19}, d1, 1, 1, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16 {%0, %1, %2, %3, %4, "
      "%5, %6, %7, %8, %9, %10, %11}, {%20, %21, %22, %23}, d2, 1, 1, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16 {%0, %1, %2, %3, %4, "
      "%5, %6, %7, %8, %9, %10, %11}, {%24, %25, %26, %27}, d3, 1, 1, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16 {%0, %1, %2, %3, %4, "
      "%5, %6, %7, %8, %9, %10, %11}, {%28, %29, %30, %31}, d4, 1, 1, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16 {%0, %1, %2, %3, %4, "
      "%5, %6, %7, %8, %9, %10, %11}, {%32, %33, %34, %35}, d5, 1, 1, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16 {%0, %1, %2, %3, %4, "
      "%5, %6, %7, %8, %9, %10, %11}, {%36, %37, %38, %39}, d6, 1, 1, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16 {%0, %1, %2, %3, %4, "
      "%5, %6, %7, %8, %9, %10, %11}, {%40, %41, %42, %43}, d7, 1, 1, 1, 0;\n}"
      : "+f"(d[0]),
        "+f"(d[1]),
        "+f"(d[2]),
        "+f"(d[3]),
        "+f"(d[4]),
        "+f"(d[5]),
        "+f"(d[6]),
        "+f"(d[7]),
        "+f"(d[8]),
        "+f"(d[9]),
        "+f"(d[10]),
        "+f"(d[11])
      : WG_A(0),
        WG_A(1),
        WG_A(2),
        WG_A(3),
        WG_A(4),
        WG_A(5),
        WG_A(6),
        WG_A(7),
        "l"(b0),
        "l"(b1),
        "r"(acc));
}
template <>
__device__ __forceinline__ void wgmma_k128<32>(
    float *d, uint32_t const (&a)[8][4], uint64_t b0, uint64_t b1, int acc) {
  asm volatile("{\n.reg .pred p;\n.reg .b64 d<8>;\nsetp.ne.b32 p, %50, 0;\n"
               "add.s64 d0, %48, 0;\nadd.s64 d1, %48, 2;\nadd.s64 d2, %48, "
               "4;\nadd.s64 d3, %48, 6;\n"
               "add.s64 d4, %49, 0;\nadd.s64 d5, %49, 2;\nadd.s64 d6, %49, "
               "4;\nadd.s64 d7, %49, 6;\n"
               "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, "
               "%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
               "{%16, %17, %18, %19}, d0, p, 1, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, "
               "%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
               "{%20, %21, %22, %23}, d1, 1, 1, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, "
               "%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
               "{%24, %25, %26, %27}, d2, 1, 1, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, "
               "%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
               "{%28, %29, %30, %31}, d3, 1, 1, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, "
               "%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
               "{%32, %33, %34, %35}, d4, 1, 1, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, "
               "%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
               "{%36, %37, %38, %39}, d5, 1, 1, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, "
               "%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
               "{%40, %41, %42, %43}, d6, 1, 1, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, "
               "%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
               "{%44, %45, %46, %47}, d7, 1, 1, 1, 0;\n}"
               : "+f"(d[0]),
                 "+f"(d[1]),
                 "+f"(d[2]),
                 "+f"(d[3]),
                 "+f"(d[4]),
                 "+f"(d[5]),
                 "+f"(d[6]),
                 "+f"(d[7]),
                 "+f"(d[8]),
                 "+f"(d[9]),
                 "+f"(d[10]),
                 "+f"(d[11]),
                 "+f"(d[12]),
                 "+f"(d[13]),
                 "+f"(d[14]),
                 "+f"(d[15])
               : WG_A(0),
                 WG_A(1),
                 WG_A(2),
                 WG_A(3),
                 WG_A(4),
                 WG_A(5),
                 WG_A(6),
                 WG_A(7),
                 "l"(b0),
                 "l"(b1),
                 "r"(acc));
}

// d (+)= A (a K half of 64: 4 K16 fragments) B (one 64-wide K half at
// descriptor b)
template <int N>
__device__ __forceinline__ void
    wgmma_k64(float *d, uint32_t const (&a)[4][4], uint64_t b, int acc);
template <>
__device__ __forceinline__ void
    wgmma_k64<8>(float *d, uint32_t const (&a)[4][4], uint64_t b, int acc) {
  asm volatile("{\n.reg .pred p;\n.reg .b64 d<4>;\nsetp.ne.b32 p, %21, 0;\n"
               "add.s64 d0, %20, 0;\nadd.s64 d1, %20, 2;\nadd.s64 d2, %20, "
               "4;\nadd.s64 d3, %20, 6;\n"
               "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, "
               "%3}, {%4, %5, %6, %7}, d0, p, 1, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, "
               "%3}, {%8, %9, %10, %11}, d1, 1, 1, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, "
               "%3}, {%12, %13, %14, %15}, d2, 1, 1, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0, %1, %2, "
               "%3}, {%16, %17, %18, %19}, d3, 1, 1, 1, 0;\n"
               "}"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : WG_A(0), WG_A(1), WG_A(2), WG_A(3), "l"(b), "r"(acc));
}
template <>
__device__ __forceinline__ void
    wgmma_k64<16>(float *d, uint32_t const (&a)[4][4], uint64_t b, int acc) {
  asm volatile(
      "{\n.reg .pred p;\n.reg .b64 d<4>;\nsetp.ne.b32 p, %25, 0;\n"
      "add.s64 d0, %24, 0;\nadd.s64 d1, %24, 2;\nadd.s64 d2, %24, 4;\nadd.s64 "
      "d3, %24, 6;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, "
      "%5, %6, %7}, {%8, %9, %10, %11}, d0, p, 1, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, "
      "%5, %6, %7}, {%12, %13, %14, %15}, d1, 1, 1, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, "
      "%5, %6, %7}, {%16, %17, %18, %19}, d2, 1, 1, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n16k16.f32.f16.f16 {%0, %1, %2, %3, %4, "
      "%5, %6, %7}, {%20, %21, %22, %23}, d3, 1, 1, 1, 0;\n"
      "}"
      : "+f"(d[0]),
        "+f"(d[1]),
        "+f"(d[2]),
        "+f"(d[3]),
        "+f"(d[4]),
        "+f"(d[5]),
        "+f"(d[6]),
        "+f"(d[7])
      : WG_A(0), WG_A(1), WG_A(2), WG_A(3), "l"(b), "r"(acc));
}
template <>
__device__ __forceinline__ void
    wgmma_k64<24>(float *d, uint32_t const (&a)[4][4], uint64_t b, int acc) {
  asm volatile(
      "{\n.reg .pred p;\n.reg .b64 d<4>;\nsetp.ne.b32 p, %29, 0;\n"
      "add.s64 d0, %28, 0;\nadd.s64 d1, %28, 2;\nadd.s64 d2, %28, 4;\nadd.s64 "
      "d3, %28, 6;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16 {%0, %1, %2, %3, %4, "
      "%5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, d0, p, 1, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16 {%0, %1, %2, %3, %4, "
      "%5, %6, %7, %8, %9, %10, %11}, {%16, %17, %18, %19}, d1, 1, 1, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16 {%0, %1, %2, %3, %4, "
      "%5, %6, %7, %8, %9, %10, %11}, {%20, %21, %22, %23}, d2, 1, 1, 1, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16 {%0, %1, %2, %3, %4, "
      "%5, %6, %7, %8, %9, %10, %11}, {%24, %25, %26, %27}, d3, 1, 1, 1, 0;\n"
      "}"
      : "+f"(d[0]),
        "+f"(d[1]),
        "+f"(d[2]),
        "+f"(d[3]),
        "+f"(d[4]),
        "+f"(d[5]),
        "+f"(d[6]),
        "+f"(d[7]),
        "+f"(d[8]),
        "+f"(d[9]),
        "+f"(d[10]),
        "+f"(d[11])
      : WG_A(0), WG_A(1), WG_A(2), WG_A(3), "l"(b), "r"(acc));
}
template <>
__device__ __forceinline__ void
    wgmma_k64<32>(float *d, uint32_t const (&a)[4][4], uint64_t b, int acc) {
  asm volatile("{\n.reg .pred p;\n.reg .b64 d<4>;\nsetp.ne.b32 p, %33, 0;\n"
               "add.s64 d0, %32, 0;\nadd.s64 d1, %32, 2;\nadd.s64 d2, %32, "
               "4;\nadd.s64 d3, %32, 6;\n"
               "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, "
               "%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
               "{%16, %17, %18, %19}, d0, p, 1, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, "
               "%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
               "{%20, %21, %22, %23}, d1, 1, 1, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, "
               "%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
               "{%24, %25, %26, %27}, d2, 1, 1, 1, 0;\n"
               "wgmma.mma_async.sync.aligned.m64n32k16.f32.f16.f16 {%0, %1, "
               "%2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, "
               "{%28, %29, %30, %31}, d3, 1, 1, 1, 0;\n"
               "}"
               : "+f"(d[0]),
                 "+f"(d[1]),
                 "+f"(d[2]),
                 "+f"(d[3]),
                 "+f"(d[4]),
                 "+f"(d[5]),
                 "+f"(d[6]),
                 "+f"(d[7]),
                 "+f"(d[8]),
                 "+f"(d[9]),
                 "+f"(d[10]),
                 "+f"(d[11]),
                 "+f"(d[12]),
                 "+f"(d[13]),
                 "+f"(d[14]),
                 "+f"(d[15])
               : WG_A(0), WG_A(1), WG_A(2), WG_A(3), "l"(b), "r"(acc));
}

struct Args {
  uint8_t const *w; // the projection's blocks [E][N / PR][K tiles][PR x 68]
  __nv_bfloat16 const *bias; // [E][N]
  int const
      *routing; // [E][T]: token t's top-k slot + 1 for expert e, 0: not routed
  __half *act;  // [E][T][I] f16: W13's output (row j: the expert's j-th routed
                // token), W2's input
  __nv_bfloat16 *eo;       // [T][TOPK][H]: W2's per-slot outputs
  float const *topk_w;     // [T][TOPK]
  __nv_bfloat16 const *x1; // [T][H] the residual
  __nv_bfloat16 *x_out;    // [T][H]
  unsigned *done; // [T][H / 64] W2 slots done (0 before; the last arriver
                  // re-zeroes it)
  float limit, alpha;
};

// GLU: W13 (128-row blocks: 64 gate rows, the 64 up rows of the same outputs;
// N = 2 I, K = H); else W2 (64-row blocks; N = H, K = I)
template <bool GLU>
struct Cfg {
  static constexpr int K = GLU ? H : I, N = GLU ? 2 * I : H;
  static constexpr int PR = GLU ? 128 : 64, MT = PR / 64, BLOCK = PR * 68,
                       KT = (K + 127) / 128;
  static constexpr int STAGES = 4;
  static constexpr int ACT_OFF = (BLOCK + 1023) / 1024 * 1024,
                       STAGE = ACT_OFF + T * 256;
  static constexpr int SMEM = STAGES * STAGE + 1024 + 16 * STAGES;
  static constexpr int ITEMS = N / PR; // a grid row: an expert's items
};

// the item: block y of expert e times its cnt routed tokens' rows (NG groups of
// 8)
template <bool GLU, int NG>
__device__ __forceinline__ void item(CUtensorMap const *amap,
                                     Args const &p,
                                     int y,
                                     int e,
                                     int cnt,
                                     int const *tokl,
                                     int const *slotl,
                                     char *sm,
                                     uint32_t full0,
                                     uint32_t empty0) {
  using C = Cfg<GLU>;
  constexpr int KT = C::KT, BLOCK = C::BLOCK, MT = C::MT, NR = 8 * NG,
                STAGES = C::STAGES;
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  uint8_t const *blocks = p.w + ((size_t)e * C::ITEMS + y) * KT * BLOCK;
  if (warp == 4) { // the producer warp: lane 0 the weights, lane j < cnt row
                   // j's activation (both K halves)
    int const src = lane < cnt ? (GLU ? tokl[lane] : e * T + lane) : 0;
    for (int kt = 0; kt < KT; kt++) {
      int const st = kt % STAGES;
      if (kt >= STAGES) {
        mbar_wait(empty0 + 8 * st, ((kt / STAGES) - 1) & 1);
      }
      uint32_t const fb = full0 + 8 * st, dst = su32(sm + st * C::STAGE);
      if (lane == 0) {
        mbar_expect(fb, BLOCK + cnt * 256);
        bulk_g2s(blocks + (size_t)kt * BLOCK, fb, dst, BLOCK);
      }
      __syncwarp();
      if (lane < cnt) {
        tma3(amap, fb, dst + C::ACT_OFF + lane * 128, 0, src, 2 * kt);
        tma3(amap,
             fb,
             dst + C::ACT_OFF + NR * 128 + lane * 128,
             0,
             src,
             2 * kt + 1);
      }
    }
    return;
  }
  int const g8 = lane >> 2, q = lane & 3,
            rbase = 16 * warp + g8; // rows rbase (+ 8) of each m64 tile
  float acc[MT][NG * 4];
  uint32_t r[2][8][4];
  auto decode = [&](char const *blk, int mt, uint32_t(&ra)[8][4]) {
#pragma unroll
    for (int rr = 0; rr < 2; rr++) {
      int const row = 64 * mt + rbase + 8 * rr;
      uint4 const v = *reinterpret_cast<uint4 const *>(blk + row * 64 + q * 16);
      uint32_t const sw =
          *reinterpret_cast<uint32_t const *>(blk + C::PR * 64 + row * 4);
      uint32_t const sc4 = (sw - 0x6C6C6C6Cu) << 2;
      uint32_t const words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int i = 0; i < 4; i++) {
        uint32_t o[4];
        decode_word(words[i], sc4, i, o);
#pragma unroll
        for (int c = 0; c < 4; c++)
          ra[2 * i + (c >> 1)][rr + 2 * (c & 1)] = o[c];
      }
    }
  };
  // W2 (one m64 tile): the K tile's two 64-wide halves are the items (two A
  // buffers in turn within the tile, as W13's two m64 tiles: a buffer
  // alternating across K tiles serializes ptxas' wgmma pipeline, C7513)
  uint32_t rh[2][4][4];
  auto decode_half = [&](char const *blk, int h, uint32_t(&ra)[4][4]) {
#pragma unroll
    for (int rr = 0; rr < 2; rr++) {
      int const row = rbase + 8 * rr;
      uint2 const v =
          *reinterpret_cast<uint2 const *>(blk + row * 64 + q * 16 + h * 8);
      uint32_t const sw =
          *reinterpret_cast<uint32_t const *>(blk + C::PR * 64 + row * 4);
      uint32_t const sc4 = (sw - 0x6C6C6C6Cu) << 2;
      uint32_t const words[2] = {v.x, v.y};
#pragma unroll
      for (int i = 0; i < 2; i++) {
        uint32_t o[4];
        decode_word(words[i], sc4, 2 * h + i, o);
#pragma unroll
        for (int c = 0; c < 4; c++)
          ra[2 * i + (c >> 1)][rr + 2 * (c & 1)] = o[c];
      }
    }
  };
  auto ktile_w2 = [&](int kt) {
    int const st = kt % STAGES;
    mbar_wait(full0 + 8 * st, (kt / STAGES) & 1);
    char const *stage = sm + st * C::STAGE;
    uint32_t const bx = su32(stage + C::ACT_OFF);
#pragma unroll
    for (int h = 0; h < 2; h++) {
      decode_half(stage, h, rh[h]);
      wgmma_fence();
      wgmma_k64<NR>(
          acc[0], rh[h], gmma_desc(bx + h * NR * 128), kt > 0 || h > 0);
      wgmma_commit();
      wgmma_wait<1>();
      if (h == 0 && kt > 0) {
        __syncwarp();
        if (lane == 0)
          mbar_arrive(empty0 + 8 * ((kt - 1) % STAGES));
      }
    }
  };
  auto ktile = [&](int kt, auto par) {
    constexpr int P = decltype(par)::value;
    int const st = kt % STAGES;
    mbar_wait(full0 + 8 * st, (kt / STAGES) & 1);
    char const *stage = sm + st * C::STAGE;
    uint32_t const bx = su32(stage + C::ACT_OFF);
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
      uint32_t(&ra)[8][4] = r[(P * MT + mt) & 1];
      decode(stage, mt, ra);
      wgmma_fence();
      wgmma_k128<NR>(
          acc[mt], ra, gmma_desc(bx), gmma_desc(bx + NR * 128), kt > 0);
      wgmma_commit();
      wgmma_wait<1>(); // every item but this one done: the other buffer free,
                       // stage kt - 1 too
      if (mt == 0 && kt > 0) {
        __syncwarp();
        if (lane == 0)
          mbar_arrive(empty0 + 8 * ((kt - 1) % STAGES));
      }
    }
  };
  if constexpr (GLU) {
#pragma unroll 1
    for (int kt = 0; kt + 1 < KT; kt += 2) {
      ktile(kt, IC<0>{});
      ktile(kt + 1, IC<1>{});
    }
    if constexpr (KT % 2) {
      ktile(KT - 1, IC<0>{});
    }
  } else {
#pragma unroll 1
    for (int kt = 0; kt < KT; kt++) {
      ktile_w2(kt);
    }
  }
  wgmma_wait<0>();
  __nv_bfloat16 const *b = p.bias + (size_t)e * C::N;
  if constexpr (GLU) {
#pragma unroll
    for (int n = 0; n < NG; n++)
#pragma unroll
      for (int i = 0; i < 4; i++) {
        int const j = 8 * n + 2 * q + (i & 1),
                  o = y * 64 + rbase + 8 * (i >> 1);
        if (j >= cnt) {
          continue;
        }
        float gate = acc[0][4 * n + i] * SUM_SCALE + __bfloat162float(b[o]);
        float up = acc[1][4 * n + i] * SUM_SCALE + __bfloat162float(b[I + o]);
        gate = fminf(gate, p.limit);
        up = fminf(fmaxf(up, -p.limit), p.limit);
        p.act[((size_t)e * T + j) * I + o] =
            __float2half((up + 1.0f) * gate / (1.0f + __expf(-gate * p.alpha)));
      }
  } else {
    // the slot outputs, then per token: the last of its TOPK experts to finish
    // this 64-row block sums them (x1 + w_k eo_k in k order, as
    // sm90_mul_sum_add) into x_out
#pragma unroll
    for (int n = 0; n < NG; n++)
#pragma unroll
      for (int i = 0; i < 4; i++) {
        int const j = 8 * n + 2 * q + (i & 1),
                  row = y * 64 + rbase + 8 * (i >> 1);
        if (j >= cnt) {
          continue;
        }
        p.eo[((size_t)tokl[j] * TOPK + slotl[j]) * H + row] = __float2bfloat16(
            acc[0][4 * n + i] * SUM_SCALE + __bfloat162float(b[row]));
      }
    __shared__ int last[T];
    __threadfence();
    asm volatile("bar.sync 1, 128;" ::: "memory");
    if (threadIdx.x < cnt) {
      int const t = tokl[threadIdx.x];
      unsigned const old = atomicAdd(p.done + t * (H / 64) + y, 1u);
      last[threadIdx.x] = old == TOPK - 1;
      if (old == TOPK - 1) {
        p.done[t * (H / 64) + y] = 0;
      }
    }
    asm volatile("bar.sync 1, 128;" ::: "memory");
    __threadfence();
    for (int x = threadIdx.x; x < cnt * 64; x += 128) {
      int const j = x >> 6, row = y * 64 + (x & 63);
      if (!last[j]) {
        continue;
      }
      int const t = tokl[j];
      float a = __bfloat162float(p.x1[(size_t)t * H + row]);
#pragma unroll
      for (int k = 0; k < TOPK; k++) {
        a += p.topk_w[t * TOPK + k] *
             __bfloat162float(__ldcg(p.eo + ((size_t)t * TOPK + k) * H + row));
      }
      p.x_out[(size_t)t * H + row] = __float2bfloat16(a);
    }
  }
}

template <bool GLU>
__global__ void __launch_bounds__(160, 1)
    moe_kernel(const __grid_constant__ CUtensorMap amap, const Args p) {
  using C = Cfg<GLU>;
  int const y = blockIdx.x, e = blockIdx.y;
  extern __shared__ __align__(1024) char smem_raw[];
  char *sm = reinterpret_cast<char *>(
      (reinterpret_cast<uintptr_t>(smem_raw) + 1023) & ~uintptr_t(1023));
  uint64_t *bars = reinterpret_cast<uint64_t *>(sm + C::STAGES * C::STAGE);
  uint32_t const full0 = su32(bars), empty0 = su32(bars + C::STAGES);
  __shared__ int tokl[T], slotl[T], s_cnt;
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  if (threadIdx.x == 0) {
    for (int s = 0; s < C::STAGES; s++) {
      mbar_init(full0 + 8 * s, 1);
      mbar_init(empty0 + 8 * s, 4);
    }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // the kernel before (the layer's gate; W13) done and visible. W13 lets W2
  // start launching (W2 reads nothing before its wait); W2 does not (the next
  // layer's kernel resets its buffers before its wait): it triggers at exit
  asm volatile("griddepcontrol.wait;" ::: "memory");
  if constexpr (GLU) {
    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
  }
  if (warp == 0) { // expert e's routed tokens, ascending
    int const rt = lane < T ? p.routing[e * T + lane] : 0;
    uint32_t const m = __ballot_sync(~0u, rt > 0);
    if (rt > 0) {
      int const j = __popc(m & ((1u << lane) - 1));
      tokl[j] = lane, slotl[j] = rt - 1;
    }
    if (lane == 0) {
      s_cnt = __popc(m);
    }
  }
  __syncthreads();
  int const cnt = s_cnt;
  if (cnt == 0) {
    return;
  }
  int const ng = (cnt + 7) / 8;
  auto go = [&](auto g) {
    item<GLU, decltype(g)::value>(
        &amap, p, y, e, cnt, tokl, slotl, sm, full0, empty0);
  };
  if (ng <= 1) {
    go(IC<1>{});
  } else if constexpr (T >= 16) {
    if (ng == 2) {
      go(IC<2>{});
    } else if constexpr (T >= 24) {
      if (ng == 3) {
        go(IC<3>{});
      } else if constexpr (T >= 32) {
        go(IC<4>{});
      }
    }
  }
}

} // namespace mk

// ---- host ----
static void map_rows(CUtensorMap *m,
                     void *base,
                     uint64_t rows,
                     uint64_t K) { // f16 [rows, K]: one row, a 64 half
  cuuint64_t gd[3] = {64, rows, K / 64};
  cuuint64_t gs[2] = {K * 2, 128};
  cuuint32_t bd[3] = {64, 1, 1}, es[3] = {1, 1, 1};
  CUresult r = cuTensorMapEncodeTiled(m,
                                      CU_TENSOR_MAP_DATA_TYPE_FLOAT16,
                                      3,
                                      base,
                                      gd,
                                      gs,
                                      bd,
                                      es,
                                      CU_TENSOR_MAP_INTERLEAVE_NONE,
                                      CU_TENSOR_MAP_SWIZZLE_128B,
                                      CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
                                      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  if (r != CUDA_SUCCESS) {
    printf("moe: tensor map error %d\n", (int)r);
  }
}

template <bool GLU>
static void launch(CUtensorMap const &m, mk::Args const &a, cudaStream_t s) {
  using C = mk::Cfg<GLU>;
  static bool init = false;
  if (!init) {
    cudaFuncSetAttribute(mk::moe_kernel<GLU>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         C::SMEM);
    init = true;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(C::ITEMS, mk::E);
  cfg.blockDim = dim3(160);
  cfg.dynamicSmemBytes = C::SMEM;
  cfg.stream = s;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  at[0].val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  cudaLaunchKernelEx(&cfg, mk::moe_kernel<GLU>, m, a);
}

struct Ctx {
  CUtensorMap m13, m2;
};

// the buffers shared by every layer: h2 [T][H] f16, act [E][T][I] f16 (their
// tensor maps)
extern "C" void *moe_ctx(void *h2, void *act) {
  Ctx *c = new Ctx;
  map_rows(&c->m13, h2, mk::T, mk::H);
  map_rows(&c->m2, act, (uint64_t)mk::E * mk::T, mk::I);
  return c;
}

extern "C" int moe_launch(void *ctx,
                          void const *routing,
                          void const *w13,
                          void const *b13,
                          void *act,
                          void const *w2,
                          void const *b2,
                          void *eo,
                          void const *topk_w,
                          void const *x1,
                          void *x_out,
                          void *done,
                          float limit,
                          float alpha,
                          void *stream) {
  Ctx *c = static_cast<Ctx *>(ctx);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  mk::Args a{(uint8_t const *)w13,
             (__nv_bfloat16 const *)b13,
             (int const *)routing,
             (__half *)act,
             (__nv_bfloat16 *)eo,
             (float const *)topk_w,
             (__nv_bfloat16 const *)x1,
             (__nv_bfloat16 *)x_out,
             (unsigned *)done,
             limit,
             alpha};
  launch<true>(c->m13, a, s);
  a.w = (uint8_t const *)w2;
  a.bias = (__nv_bfloat16 const *)b2;
  launch<false>(c->m2, a, s);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    printf("moe_launch: %s\n", cudaGetErrorString(e));
    return 1;
  }
  return 0;
}
