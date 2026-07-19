// =============================================================================
// FUSED DeepSeek-V3 dense-MLP decode megakernel (TP8 EP2 per-rank, bs=1, M=1).
//
// Replaces the unfused 3-task MPK dense chain (rmsnorm+quant -> W13 GEMV ->
// silu_mul(384-chunk) + requant -> W2 GEMV -> cast) with ONE fused kernel.
//
// Reuses the proven device primitives from
//   include/mirage/persistent_kernel/tasks/blackwell/ffn_full_megakernel_sm100.cuh
// (grid barrier, cp.async helpers, fp8 dequant, UE8M0 quant_scale, the packed
// half2 GEMV dgemv_cpa16_h2, the 4B GEMV dgemv_cpa, RMSNorm + quant_group_warp)
// — adapted to the DENSE shapes (W13_N=4608, W2_K=2304) + the 384-chunk silu
// interleave + single-MLP (no router / topk / EP filter / shared-expert).
//
// CRITICAL CONTRACT NOTES (the frozen gate enforces these):
//  * NO __ldg / ld.global.nc anywhere (per-step buffers stale across decode
//    steps in the megakernel -> token flip). The in-tree dgemv_* use
//    __ldg(&w_scale_row[g]); HERE that is a PLAIN load.
//  * extern __shared__ __align__(1024) (megakernel smem convention).
//  * grid.x == NUM_WORKERS == 136 (barrier participants).
//  * thread partition DERIVED from blockDim.x (256 production / 512 contract).
//  * ACTIVATION scales UE8M0 (quant_scale); WEIGHT scales RAW float32
//  [n>>7][k>>7].
//  * silu interleave = 384 (6 chunk-pairs):
//  out[c]=silu(y13[cp*768+wc])*y13[cp*768+384+wc].
// =============================================================================
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

// The harness provides mirage::runtime::{TaskDesc,RuntimeConfig} before it
// #includes this header. We forward-declare nothing; we use those types.

// ---- MPK grid barrier (semantics VERBATIM from ffn_full_megakernel_sm100.cuh)
// -
struct DenseMlpGridBarrier {
  unsigned int *count; // [1] arrivals in the current generation
  unsigned int *gen;   // [1] generation (sense) counter
};

// Block-collective double-fence barrier; only thread 0 touches global mem.
__device__ __forceinline__ void dense_mlp_grid_barrier(DenseMlpGridBarrier b,
                                                       int num_participants) {
  __syncthreads();
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    unsigned int my_gen = *((unsigned int volatile *)b.gen);
    unsigned int prev = atomicAdd(b.count, 1u);
    if (prev + 1u == (unsigned int)num_participants) {
      *b.count = 0u;
      __threadfence();
      atomicAdd(b.gen, 1u);
    } else {
      while (*((unsigned int volatile *)b.gen) == my_gen) { /* spin */
      }
    }
  }
  __syncthreads();
}

namespace kernel {
namespace dsv3_dense_mlp {

// ---- Problem shapes (DSv3 TP8 EP2 per-rank DENSE MLP) -----------------------
static constexpr int HIDDEN = 7168;   // W13 K, W2 N
static constexpr int W13_N = 4608;    // gate+up output width
static constexpr int W2_K = 2304;     // silu output width / W2 K
static constexpr int SILU_OUT = 2304; // = W2_K
static constexpr int CHUNK = 384;     // silu interleave chunk (NOT 512)
static constexpr int N_CHUNK_PAIRS = 6;
static constexpr int GRP = 128;
static constexpr int KG1 = HIDDEN / GRP; // 56 (W13 K-groups)
static constexpr int NB1 = W13_N / GRP;  // 36 (W13 N-blocks)
static constexpr int KG2 = W2_K / GRP;   // 18 (W2 K-groups)
static constexpr int NB2 = HIDDEN / GRP; // 56 (W2 N-blocks)
static constexpr int NUM_WORKERS =
    136; // B200 worker pool (barrier participants)
static constexpr float RMS_EPS = 1e-6f;

// ============================================================================
//  Per-task Scratch (block-0 globals + the cross-phase barrier).
//  Layout: [barrier(8B)+pad to 64] [y13(4608 f32)]. The only GLOBAL cross-block
//  buffer is y13 (W13 writes it from many blocks, every block reads it back in
//  Phase 2). rmsnorm/a_fp8/a_scale/i_fp8/i_scale are block-local (recomputed
//  redundantly per block, like the FFN front stages) so no extra global region
//  or barrier is needed for them.
// ============================================================================
static constexpr int BARRIER_BYTES = 2 * (int)sizeof(uint32_t);

struct Scratch {
  float *y13; // [W13_N] fp32 (the only cross-block global)
};

__device__ __forceinline__ Scratch make_scratch(uint8_t *base) {
  Scratch sc;
  size_t off = 64; // first 64B reserved for the barrier (count+gen) + pad
  sc.y13 = reinterpret_cast<float *>(base + off);
  return sc;
}

// ---- bit helpers ------------------------------------------------------------
__device__ __forceinline__ uint32_t dm_f2u(float f) {
  return __float_as_uint(f);
}
__device__ __forceinline__ float dm_u2f(uint32_t u) {
  return __uint_as_float(u);
}
__device__ __forceinline__ uint8_t encode_ue8m0(float scale) {
  scale = fmaxf(scale, 1e-30f);
  uint32_t bits = dm_f2u(scale);
  int exp_unbiased = (int)((bits >> 23) & 0xFF) - 127;
  uint32_t mantissa = bits & 0x7FFFFF;
  int ue = (mantissa == 0 ? exp_unbiased : exp_unbiased + 1) + 127;
  ue = ue < 0 ? 0 : ue;
  ue = ue > 255 ? 255 : ue;
  return (uint8_t)ue;
}
__device__ __forceinline__ float decode_ue8m0(uint8_t e) {
  return dm_u2f((uint32_t)e << 23);
}
__device__ __forceinline__ float quant_scale(float amax) {
  return decode_ue8m0(encode_ue8m0(amax / 448.0f));
}
__device__ __forceinline__ float f8(uint8_t x) {
  return __half2float(__nv_cvt_fp8_to_halfraw(
      *reinterpret_cast<__nv_fp8_storage_t *>(&x), __NV_E4M3));
}
__device__ __forceinline__ uint8_t to_f8(float v) {
  __nv_fp8_storage_t s = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
  return *reinterpret_cast<uint8_t *>(&s);
}
__device__ __forceinline__ float silu(float x) {
  return x / (1.0f + expf(-x));
}

// FP8(e4m3)->__half2 path (cvt.rn.f16x2.e4m3x2). e4m3 reaches 448 so raw fp16
// products would overflow; the GEMV pre-scales activations by 2^-8 (exact exp
// shift) and multiplies the dot back by 256.
__device__ __forceinline__ __half2 dm_f8x2_h2(uint32_t v16) {
  __half2_raw r = __nv_cvt_fp8x2_to_halfraw2(
      (__nv_fp8x2_storage_t)(v16 & 0xffff), __NV_E4M3);
  return *reinterpret_cast<__half2 *>(&r);
}
__device__ __forceinline__ void
    dm_f8x4_h2(uint32_t v, __half2 &b0, __half2 &b1) {
  b0 = dm_f8x2_h2(v & 0xffff);
  b1 = dm_f8x2_h2((v >> 16) & 0xffff);
}
__device__ __forceinline__ void dm_f8x16_h2(uint4 v, __half2 b[8]) {
  dm_f8x4_h2(v.x, b[0], b[1]);
  dm_f8x4_h2(v.y, b[2], b[3]);
  dm_f8x4_h2(v.z, b[4], b[5]);
  dm_f8x4_h2(v.w, b[6], b[7]);
}
__device__ __forceinline__ void
    f8x4(uint32_t v, float &f0, float &f1, float &f2, float &f3) {
  __half2_raw lo =
      __nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t)(v & 0xffff), __NV_E4M3);
  __half2_raw hi = __nv_cvt_fp8x2_to_halfraw2(
      (__nv_fp8x2_storage_t)((v >> 16) & 0xffff), __NV_E4M3);
  f0 = __half2float(*(__half *)&lo.x);
  f1 = __half2float(*(__half *)&lo.y);
  f2 = __half2float(*(__half *)&hi.x);
  f3 = __half2float(*(__half *)&hi.y);
}

// ---- per-128-group warp quantizer (UE8M0 scale). src is bf16/float. ---------
template <typename T>
__device__ __forceinline__ void
    quant_group_warp(T const *src, uint8_t *q, float *scale, int g, int lane) {
  float v[4], amax = 0.f;
#pragma unroll
  for (int t = 0; t < 4; t++) {
    float x = (float)src[g * GRP + lane * 4 + t];
    v[t] = x;
    amax = fmaxf(amax, fabsf(x));
  }
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
  }
  float s = quant_scale(amax);
  float inv = 1.f / s;
  if (lane == 0) {
    scale[g] = s;
  }
#pragma unroll
  for (int t = 0; t < 4; t++) {
    q[g * GRP + lane * 4 + t] = to_f8(v[t] * inv);
  }
}

// ---- cp.async helpers (VERBATIM from the COLD FFN) --------------------------
__device__ __forceinline__ void cpasync4(uint32_t smem_addr, void const *gptr) {
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(smem_addr),
               "l"(gptr));
}
__device__ __forceinline__ void cpasync16(uint32_t smem_addr,
                                          void const *gptr) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_addr),
               "l"(gptr));
}
__device__ __forceinline__ void cpasync_commit() {
  asm volatile("cp.async.commit_group;\n");
}
template <int N>
__device__ __forceinline__ void cpasync_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}

// ============================================================================
//  SCALAR fp32 uint4 (16B) cp.async.cg pipelined GEMV — W13 path (K=7168,
//  K%512==0). One warp computes RBX consecutive output rows. Full per-element
//  fp32 dequant + fp32 accumulation (matches the reference's accumulation more
//  tightly than the half2 path -> keeps the final bf16 max_abs under 0.05).
//  Plain weight-scale load (NO __ldg).
// ============================================================================
template <int RBX, int STAGES>
__device__ __forceinline__ void
    dgemv_cpa16(uint8_t const *__restrict__ a_fp8,
                float const *__restrict__ a_scale,
                uint8_t const *__restrict__ w_fp8,
                float const *__restrict__ w_scale_row,
                int K,
                int n0,
                int lane,
                uint4 *wbuf_base,
                float *y_out) {
  uint4 const *a16 = reinterpret_cast<uint4 const *>(a_fp8);
  uint4 const *w16 = reinterpret_cast<uint4 const *>(w_fp8 + (size_t)n0 * K);
  int KUr = K >> 4;
  int SS = KUr >> 5;
  float y[RBX];
#pragma unroll
  for (int r = 0; r < RBX; r++) {
    y[r] = 0.f;
  }
  uint32_t const sbase = __cvta_generic_to_shared(wbuf_base);
  uint32_t const STRIDE = (uint32_t)(RBX * 32 * 16);
  int pf = (STAGES - 1 < SS) ? (STAGES - 1) : SS;
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < pf) {
      uint32_t b = sbase + (uint32_t)s * STRIDE;
#pragma unroll
      for (int r = 0; r < RBX; r++) {
        cpasync16(b + (uint32_t)((r * 32 + lane) * 16),
                  &w16[(size_t)r * KUr + (size_t)s * 32 + lane]);
      }
    }
    cpasync_commit();
  }
  for (int ss = 0; ss < SS; ss++) {
    int sp = ss + (STAGES - 1);
    if (sp < SS) {
      uint32_t b = sbase + (uint32_t)(sp % STAGES) * STRIDE;
#pragma unroll
      for (int r = 0; r < RBX; r++) {
        cpasync16(b + (uint32_t)((r * 32 + lane) * 16),
                  &w16[(size_t)r * KUr + (size_t)sp * 32 + lane]);
      }
    }
    cpasync_commit();
    cpasync_wait<STAGES - 1>();
    __syncwarp();
    int g = (ss * 32 + lane) >> 3;
    float sc = a_scale[g] * w_scale_row[g];
    uint4 av = a16[ss * 32 + lane];
    float a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15;
    f8x4(av.x, a0, a1, a2, a3);
    f8x4(av.y, a4, a5, a6, a7);
    f8x4(av.z, a8, a9, a10, a11);
    f8x4(av.w, a12, a13, a14, a15);
    uint32_t cur = sbase + (uint32_t)(ss % STAGES) * STRIDE;
#pragma unroll
    for (int r = 0; r < RBX; r++) {
      uint4 wv;
      uint32_t saddr = cur + (uint32_t)((r * 32 + lane) * 16);
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(wv.x), "=r"(wv.y), "=r"(wv.z), "=r"(wv.w)
                   : "r"(saddr));
      float w0, w1, w2, w3, w4_, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14,
          w15;
      f8x4(wv.x, w0, w1, w2, w3);
      f8x4(wv.y, w4_, w5, w6, w7);
      f8x4(wv.z, w8, w9, w10, w11);
      f8x4(wv.w, w12, w13, w14, w15);
      float acc = a0 * w0;
      acc += a1 * w1;
      acc += a2 * w2;
      acc += a3 * w3;
      acc += a4 * w4_;
      acc += a5 * w5;
      acc += a6 * w6;
      acc += a7 * w7;
      acc += a8 * w8;
      acc += a9 * w9;
      acc += a10 * w10;
      acc += a11 * w11;
      acc += a12 * w12;
      acc += a13 * w13;
      acc += a14 * w14;
      acc += a15 * w15;
      y[r] += acc * sc;
    }
  }
#pragma unroll
  for (int r = 0; r < RBX; r++) {
    float v = y[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, o);
    }
    if (lane == 0) {
      y_out[r] = v;
    }
  }
}

// ============================================================================
//  PACKED-half2 uint4 (16B) cp.async.cg pipelined GEMV — W13 path (K=7168,
//  K%512==0). One warp computes RBX consecutive output rows. NOTE: the in-tree
//  source loads the weight-scale row via __ldg; HERE we use a plain load (the
//  gate bans __ldg on the per-step weight-scale buffer).
// ============================================================================
template <int RBX, int STAGES>
__device__ __forceinline__ void
    dgemv_cpa16_h2(uint8_t const *__restrict__ a_fp8,
                   float const *__restrict__ a_scale,
                   uint8_t const *__restrict__ w_fp8,
                   float const *__restrict__ w_scale_row,
                   int K,
                   int n0,
                   int lane,
                   uint4 *wbuf_base,
                   float *y_out) {
  uint4 const *a16 = reinterpret_cast<uint4 const *>(a_fp8);
  uint4 const *w16 = reinterpret_cast<uint4 const *>(w_fp8 + (size_t)n0 * K);
  int KUr = K >> 4;  // uint4 per row (row stride)
  int SS = KUr >> 5; // super-steps (32 lanes * uint4 = 512 fp8)
  const __half2 kInvScale = __float2half2_rn(0.00390625f); // 2^-8
  float y[RBX];
#pragma unroll
  for (int r = 0; r < RBX; r++) {
    y[r] = 0.f;
  }
  uint32_t const sbase = __cvta_generic_to_shared(wbuf_base);
  uint32_t const STRIDE = (uint32_t)(RBX * 32 * 16);
  int pf = (STAGES - 1 < SS) ? (STAGES - 1) : SS;
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < pf) {
      uint32_t b = sbase + (uint32_t)s * STRIDE;
#pragma unroll
      for (int r = 0; r < RBX; r++) {
        cpasync16(b + (uint32_t)((r * 32 + lane) * 16),
                  &w16[(size_t)r * KUr + (size_t)s * 32 + lane]);
      }
    }
    cpasync_commit();
  }
  for (int ss = 0; ss < SS; ss++) {
    int sp = ss + (STAGES - 1);
    if (sp < SS) {
      uint32_t b = sbase + (uint32_t)(sp % STAGES) * STRIDE;
#pragma unroll
      for (int r = 0; r < RBX; r++) {
        cpasync16(b + (uint32_t)((r * 32 + lane) * 16),
                  &w16[(size_t)r * KUr + (size_t)sp * 32 + lane]);
      }
    }
    cpasync_commit();
    cpasync_wait<STAGES - 1>();
    __syncwarp();
    int g = (ss * 32 + lane) >> 3;
    float sc = a_scale[g] * w_scale_row[g] * 256.0f; // undo 2^-8 prescale
    uint4 av = a16[ss * 32 + lane];
    __half2 ah[8];
    dm_f8x16_h2(av, ah);
#pragma unroll
    for (int i = 0; i < 8; i++) {
      ah[i] = __hmul2(ah[i], kInvScale);
    }
    uint32_t cur = sbase + (uint32_t)(ss % STAGES) * STRIDE;
#pragma unroll
    for (int r = 0; r < RBX; r++) {
      uint4 wv;
      uint32_t saddr = cur + (uint32_t)((r * 32 + lane) * 16);
      asm volatile("ld.shared.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                   : "=r"(wv.x), "=r"(wv.y), "=r"(wv.z), "=r"(wv.w)
                   : "r"(saddr));
      __half2 bw[8];
      dm_f8x16_h2(wv, bw);
      __half2 p = __hmul2(ah[0], bw[0]);
      p = __hfma2(ah[1], bw[1], p);
      p = __hfma2(ah[2], bw[2], p);
      p = __hfma2(ah[3], bw[3], p);
      p = __hfma2(ah[4], bw[4], p);
      p = __hfma2(ah[5], bw[5], p);
      p = __hfma2(ah[6], bw[6], p);
      p = __hfma2(ah[7], bw[7], p);
      float acc = __half2float(p.x) + __half2float(p.y);
      y[r] += acc * sc;
    }
  }
#pragma unroll
  for (int r = 0; r < RBX; r++) {
    float v = y[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, o);
    }
    if (lane == 0) {
      y_out[r] = v;
    }
  }
}

// ============================================================================
//  4-byte cp.async pipelined GEMV — W2 path (K=2304, K%512=256 != 0 -> the
//  half2 16B path would DROP the last 256 cols, so this 4B path is MANDATORY).
//  KG = K/128 quant groups (18 for W2). Plain weight-scale load (NO __ldg).
// ============================================================================
template <int RBX, int STAGES>
__device__ __forceinline__ void dgemv_cpa(uint8_t const *__restrict__ a_fp8,
                                          float const *__restrict__ a_scale,
                                          uint8_t const *__restrict__ w_fp8,
                                          float const *__restrict__ w_scale_row,
                                          int K,
                                          int KG,
                                          int n0,
                                          int lane,
                                          uint32_t *wbuf_base,
                                          float *y_out) {
  uint32_t const *a4 = reinterpret_cast<uint32_t const *>(a_fp8);
  uint32_t const *w4 =
      reinterpret_cast<uint32_t const *>(w_fp8 + (size_t)n0 * K);
  int Kw = K / 4;
  float y[RBX];
#pragma unroll
  for (int r = 0; r < RBX; r++) {
    y[r] = 0.f;
  }
  uint32_t const sbase = __cvta_generic_to_shared(wbuf_base);
  uint32_t const STRIDE = (uint32_t)(RBX * 32 * 4);
  int pf = (STAGES - 1 < KG) ? (STAGES - 1) : KG;
#pragma unroll
  for (int s = 0; s < STAGES - 1; s++) {
    if (s < pf) {
      uint32_t b = sbase + (uint32_t)s * STRIDE;
#pragma unroll
      for (int r = 0; r < RBX; r++) {
        cpasync4(b + (uint32_t)((r * 32 + lane) * 4),
                 &w4[(size_t)r * Kw + (size_t)s * 32 + lane]);
      }
    }
    cpasync_commit();
  }
  for (int g = 0; g < KG; g++) {
    int gp = g + (STAGES - 1);
    if (gp < KG) {
      uint32_t b = sbase + (uint32_t)(gp % STAGES) * STRIDE;
#pragma unroll
      for (int r = 0; r < RBX; r++) {
        cpasync4(b + (uint32_t)((r * 32 + lane) * 4),
                 &w4[(size_t)r * Kw + (size_t)gp * 32 + lane]);
      }
    }
    cpasync_commit();
    cpasync_wait<STAGES - 1>();
    __syncwarp();
    uint32_t cur = sbase + (uint32_t)(g % STAGES) * STRIDE;
    uint32_t av = a4[g * 32 + lane];
    float sc = a_scale[g] * w_scale_row[g];
    float a0, a1, a2, a3;
    f8x4(av, a0, a1, a2, a3);
#pragma unroll
    for (int r = 0; r < RBX; r++) {
      uint32_t w;
      uint32_t saddr = cur + (uint32_t)((r * 32 + lane) * 4);
      asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(w) : "r"(saddr));
      float w0, w1, w2, w3;
      f8x4(w, w0, w1, w2, w3);
      float acc = a0 * w0;
      acc += a1 * w1;
      acc += a2 * w2;
      acc += a3 * w3;
      y[r] += acc * sc;
    }
  }
#pragma unroll
  for (int r = 0; r < RBX; r++) {
    float v = y[r];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      v += __shfl_down_sync(0xffffffffu, v, o);
    }
    if (lane == 0) {
      y_out[r] = v;
    }
  }
}

// ============================================================================
//  THE FUSED TASK.  ABI mandated by gate/harness.cu.
//  __launch_bounds__(256,1) — MPK production worker is 256 threads.
// ============================================================================
__device__ __noinline__ void dsv3_dense_mlp_fused_task_impl(
    mirage::runtime::TaskDesc const *task_desc,
    int merge_task_offset,
    mirage::runtime::RuntimeConfig const &runtime_config) {
  (void)runtime_config;

  __nv_bfloat16 const *hidden =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[0]);
  uint8_t const *w13 = static_cast<uint8_t const *>(task_desc->input_ptrs[1]);
  float const *w13_scale = static_cast<float const *>(task_desc->input_ptrs[2]);
  uint8_t const *w2 = static_cast<uint8_t const *>(task_desc->input_ptrs[3]);
  float const *w2_scale = static_cast<float const *>(task_desc->input_ptrs[4]);
  __nv_bfloat16 const *rmsnorm_weight =
      static_cast<__nv_bfloat16 const *>(task_desc->input_ptrs[5]);
  uint8_t *scratch_base = static_cast<uint8_t *>(task_desc->input_ptrs[6]);

  __nv_bfloat16 *out = static_cast<__nv_bfloat16 *>(task_desc->output_ptrs[0]);
#ifdef GATE_DEBUG_TAPS
  float *dbg_rmsnorm = static_cast<float *>(task_desc->output_ptrs[1]);
  float *dbg_w13 = static_cast<float *>(task_desc->output_ptrs[2]);
  float *dbg_silu = static_cast<float *>(task_desc->output_ptrs[3]);
  float *dbg_ascale = static_cast<float *>(task_desc->output_ptrs[4]);
  float *dbg_iscale = static_cast<float *>(task_desc->output_ptrs[5]);
  uint8_t *dbg_afp8 = static_cast<uint8_t *>(task_desc->output_ptrs[6]);
  uint8_t *dbg_ifp8 = static_cast<uint8_t *>(task_desc->output_ptrs[7]);
#endif

  DenseMlpGridBarrier barrier;
  barrier.count = reinterpret_cast<unsigned int *>(scratch_base);
  barrier.gen =
      reinterpret_cast<unsigned int *>(scratch_base + sizeof(uint32_t));
  Scratch sc = make_scratch(scratch_base);

  int const worker_idx = merge_task_offset;
  int const TPB = (int)blockDim.x; // 256 production / 512 contract
  int const gtid = worker_idx * TPB + threadIdx.x;
  int const gthreads = NUM_WORKERS * TPB;
  int const gwarp = gtid / 32;
  int const lane = threadIdx.x & 31;
  int const gwarps = gthreads / 32;
  int const wlocal = threadIdx.x >> 5; // within-worker warp id
  int const nwl = TPB >> 5;            // warps per worker

  // GEMV row-block / pipeline stages. RBX_* must divide GRP=128 (the shared
  // per-N-block weight-scale row is keyed by n0/GRP and shared by all RBX
  // rows).
  constexpr int RBX_W13 = 8;
  constexpr int RBX_W2 = 16;
  constexpr int ST_W13 = 4;
  constexpr int ST_W2 = 3;
  static_assert((128 % RBX_W13) == 0 && (128 % RBX_W2) == 0,
                "RBX_* must divide GRP=128 (shared scale-row validity)");

  // Per-warp cp.async weight stage buffer (worst case across W13 16B / W2 4B).
  // W13: RBX_W13*32*ST_W13 uint4 ; W2: ceil(RBX_W2*32*ST_W2 / 4) uint4.
  constexpr size_t U4_W13 = (size_t)RBX_W13 * 32 * ST_W13;        // 8192
  constexpr size_t U4_W2 = ((size_t)RBX_W2 * 32 * ST_W2 + 3) / 4; // 384
  constexpr size_t WBUF_U4 = U4_W13 > U4_W2 ? U4_W13 : U4_W2;

  extern __shared__ __align__(1024) uint8_t s_smem[];
#define DM_AU16(x) (((x) + 15u) & ~((size_t)15u))
  size_t off = 0;
  uint4 *s_wbuf = reinterpret_cast<uint4 *>(s_smem + off);
  off += (size_t)nwl * WBUF_U4 * sizeof(uint4);
  uint4 *my_wbuf = s_wbuf + (size_t)wlocal * WBUF_U4;
  uint32_t *my_wbuf4 = reinterpret_cast<uint32_t *>(my_wbuf);
  off = DM_AU16(off);
  __nv_bfloat16 *s_norm = reinterpret_cast<__nv_bfloat16 *>(s_smem + off);
  off += (size_t)HIDDEN * sizeof(__nv_bfloat16); // 14336
  off = DM_AU16(off);
  uint8_t *s_a = s_smem + off; // 16-aligned (dgemv uint4 activation read)
  off += HIDDEN;               // 7168
  off = DM_AU16(off);
  float *s_as = reinterpret_cast<float *>(s_smem + off);
  off += (size_t)KG1 * sizeof(float); // 224
  off = DM_AU16(off);
  // silu intermediate + W2 input (block-local; recomputed by every block).
  float *s_silu = reinterpret_cast<float *>(s_smem + off);
  off += (size_t)SILU_OUT * sizeof(float); // 9216
  off = DM_AU16(off);
  uint8_t *s_ifp8 = s_smem + off; // 16-aligned (block-local W2 input)
  off += (size_t)W2_K;            // 2304
  off = DM_AU16(off);
  float *s_iscale = reinterpret_cast<float *>(s_smem + off);
  off += (size_t)KG2 * sizeof(float); // 72
  off = DM_AU16(off);
  // per-warp RMSNorm reduction scratch.
  float *s_red = reinterpret_cast<float *>(s_smem + off);
  off += (size_t)nwl * sizeof(float);
#undef DM_AU16

  // ====================================================================
  //  Phase A — RMSNorm. Every block computes the FULL sum(x^2) over HIDDEN
  //  redundantly, then writes the bf16 normed into block-local s_norm.
  //  normed[i] = bf16(x[i] * rms_rcp * w[i]).
  // ====================================================================
  {
    float ss = 0.f;
    for (int i = threadIdx.x; i < HIDDEN; i += TPB) {
      float v = __bfloat162float(hidden[i]);
      ss += v * v;
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      ss += __shfl_xor_sync(0xffffffffu, ss, o);
    }
    if (lane == 0) {
      s_red[wlocal] = ss;
    }
    __syncthreads();
    float tot = (threadIdx.x < nwl) ? s_red[threadIdx.x] : 0.f;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      tot += __shfl_xor_sync(0xffffffffu, tot, o);
    }
    if (threadIdx.x == 0) {
      s_red[0] = tot;
    }
    __syncthreads();
    float rms_rcp = rsqrtf(s_red[0] / float(HIDDEN) + RMS_EPS);
    for (int i = threadIdx.x; i < HIDDEN; i += TPB) {
      float v = __bfloat162float(hidden[i]);
      float wt = __bfloat162float(rmsnorm_weight[i]);
      s_norm[i] = __float2bfloat16(v * rms_rcp * wt);
    }
    __syncthreads();
#ifdef GATE_DEBUG_TAPS
    if (blockIdx.x == 0) {
      for (int i = threadIdx.x; i < HIDDEN; i += TPB) {
        dbg_rmsnorm[i] = __bfloat162float(s_norm[i]);
      }
    }
#endif
  }

  // ====================================================================
  //  Phase 0 — FP8-quantize the BF16 normed (UE8M0 scale) into block-local
  //  s_a / s_as. blk0 persists debug taps.
  // ====================================================================
  for (int g = wlocal; g < KG1; g += nwl) {
    quant_group_warp<__nv_bfloat16>(s_norm, s_a, s_as, g, lane);
  }
  __syncthreads();
#ifdef GATE_DEBUG_TAPS
  if (blockIdx.x == 0) {
    for (int i = threadIdx.x; i < HIDDEN; i += TPB) {
      dbg_afp8[i] = s_a[i];
    }
    for (int g = threadIdx.x; g < KG1; g += TPB) {
      dbg_ascale[g] = s_as[g];
    }
  }
#endif

  // ====================================================================
  //  Phase 1 — W13 GEMV over ALL 4608 rows -> global sc.y13[n].
  //  4608/RBX_W13 = 576 warp-jobs distributed over gwarps grid warps.
  // ====================================================================
  int const n13 = W13_N / RBX_W13; // 576
  for (int idx = gwarp; idx < n13; idx += gwarps) {
    int n0 = idx * RBX_W13;
    float const *ws = w13_scale + (size_t)(n0 / GRP) * KG1;
    float yb[RBX_W13];
    dgemv_cpa16<RBX_W13, ST_W13>(
        s_a, s_as, w13, ws, HIDDEN, n0, lane, my_wbuf, yb);
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < RBX_W13; r++) {
        sc.y13[n0 + r] = yb[r];
      }
    }
  }
  dense_mlp_grid_barrier(barrier, NUM_WORKERS);

#ifdef GATE_DEBUG_TAPS
  if (blockIdx.x == 0) {
    for (int n = threadIdx.x; n < W13_N; n += TPB) {
      dbg_w13[n] = sc.y13[n];
    }
  }
#endif

  // ====================================================================
  //  Phase 2 — silu_mul (384-chunk interleave) + UE8M0 requant.
  //  Every block recomputes the full silu_out from the global y13 into
  //  block-local s_silu, then requants -> block-local s_ifp8 / s_iscale, so
  //  Phase 3 reads the W2 input from SMEM (no cold y13 global re-read in the
  //  GEMV) and no Phase2->3 grid barrier is needed.
  //
  //  384-interleave: out[c] = silu(y13[cp*768 + wc]) * y13[cp*768 + 384 + wc]
  //  where cp = c/384, wc = c%384. (NOT the MoE 512-layout y[i] vs y[512+i].)
  // ====================================================================
  for (int c = threadIdx.x; c < SILU_OUT; c += TPB) {
    int cp = c / CHUNK;
    int wc = c % CHUNK;
    float gate = sc.y13[cp * 768 + wc];
    float up = sc.y13[cp * 768 + 384 + wc];
    s_silu[c] = silu(gate) * up;
  }
  __syncthreads();
#ifdef GATE_DEBUG_TAPS
  if (blockIdx.x == 0) {
    for (int c = threadIdx.x; c < SILU_OUT; c += TPB) {
      dbg_silu[c] = s_silu[c];
    }
  }
#endif
  // requant silu_out (per-128-group UE8M0) into block-local s_ifp8 / s_iscale.
  for (int g = wlocal; g < KG2; g += nwl) {
    quant_group_warp<float>(s_silu, s_ifp8, s_iscale, g, lane);
  }
  __syncthreads();
#ifdef GATE_DEBUG_TAPS
  if (blockIdx.x == 0) {
    for (int i = threadIdx.x; i < W2_K; i += TPB) {
      dbg_ifp8[i] = s_ifp8[i];
    }
    for (int g = threadIdx.x; g < KG2; g += TPB) {
      dbg_iscale[g] = s_iscale[g];
    }
  }
#endif

  // ====================================================================
  //  Phase 3 — W2 GEMV over ALL 7168 rows -> out (bf16). 4B path (K=2304
  //  not %512). 7168/RBX_W2 = 448 warp-jobs over gwarps grid warps.
  // ====================================================================
  int const n2 = HIDDEN / RBX_W2; // 448
  for (int idx = gwarp; idx < n2; idx += gwarps) {
    int n0 = idx * RBX_W2;
    float const *ws = w2_scale + (size_t)(n0 / GRP) * KG2;
    float yb[RBX_W2];
    dgemv_cpa<RBX_W2, ST_W2>(
        s_ifp8, s_iscale, w2, ws, W2_K, KG2, n0, lane, my_wbuf4, yb);
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < RBX_W2; r++) {
        out[n0 + r] = __float2bfloat16(yb[r]);
      }
    }
  }
}

} // namespace dsv3_dense_mlp
} // namespace kernel
