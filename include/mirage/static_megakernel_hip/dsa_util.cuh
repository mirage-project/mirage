// dsa_util.cuh (HIP, gfx950) -- helpers of GLM-5.3-Flash's DSA tasks on AMD (tasks/dsa_*.cuh of this directory), the
// math of include/mirage/static_megakernel/dsa_common.cuh with SGLang ROCm's layouts
// (amd/notes/sglang_v0522_glm_dsa_contract.md). Not a task.
//   FP8 (indexer)  e4m3 with one power-of-two (UE8M0) scale per 128 values: s = 2^ceil(log2(max(amax, 1e-4) / 448)),
//                  q = e4m3(x / s) (SGLang act_quant / kpool _quant_fp8_ue8m0)
//   pooled page    SGLang ROCm's preshuffled index-K page of 64 pooled keys (kpool_fp8_index.py, tile 16): key o, dim c at
//                  byte (o / 16) 2048 + (c / 16) 256 + (o % 16) 16 + c % 16; its fp32 scale at byte 8192 + 4 o; 8448 B
// The vector helpers use 32 lanes of one wave (lane l < 32 holds dims 4 l .. 4 l + 3 of a 128-vector; 32-wide shuffles).
#pragma once
#include "core.cuh"

namespace static_mk {
namespace dsa {

constexpr int IDX_PAGE_KEYS = 64;                      // pooled keys in a page
constexpr int IDX_PAGE_BYTES = IDX_PAGE_KEYS * (128 + 4); // 8448
constexpr int IDX_SCALE_OFF = IDX_PAGE_KEYS * 128;        // 8192

// byte offset of key o (< 64), dims c .. c + 3 (c % 4 == 0) in a preshuffled pooled page
__host__ __device__ constexpr int idx_key_off(int o, int c) {
  return (o / 16) * 2048 + (c / 16) * 256 + (o % 16) * 16 + c % 16;
}

// the UE8M0 exponent byte e of a group with absolute maximum amax (scale 2^(e - 127))
__device__ __forceinline__ int ue8m0_exp(float amax) {
  float const r = __fdiv_rn(fmaxf(amax, 1e-4f), 448.0f);
  return ceil_log2_bits(r) + 127;
}
// 2^(e - 127) as fp32 (e in 1 .. 254)
__device__ __forceinline__ float exp2_int(int e) {
  return __uint_as_float((uint32_t)e << 23);
}

// 32 lanes: e4m3 with a UE8M0 scale of N 128-vectors (lane l: dims 4 l .. + 3 of vector j in v[j]); q[j] the lane's 4
// bytes, e[j] the exponent byte (every lane)
template <int N>
__device__ __forceinline__ void quant128(float const (&v)[N][4], uint32_t (&q)[N], int (&e)[N]) {
  float amax[N];
#pragma unroll
  for (int j = 0; j < N; j++) {
    amax[j] = fmaxf(fmaxf(fabsf(v[j][0]), fabsf(v[j][1])), fmaxf(fabsf(v[j][2]), fabsf(v[j][3])));
  }
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) {
#pragma unroll
    for (int j = 0; j < N; j++) {
      amax[j] = fmaxf(amax[j], __shfl_xor(amax[j], d, 32));
    }
  }
#pragma unroll
  for (int j = 0; j < N; j++) {
    e[j] = ue8m0_exp(amax[j]);
    float const inv = exp2_int(254 - e[j]); // 1 / s, exact
    q[j] = (uint32_t)cvt_e4m3(v[j][0] * inv) | ((uint32_t)cvt_e4m3(v[j][1] * inv) << 8) |
           ((uint32_t)cvt_e4m3(v[j][2] * inv) << 16) | ((uint32_t)cvt_e4m3(v[j][3] * inv) << 24);
  }
}

// 32 lanes: the normalized Hadamard transform of N 128-vectors in SGLang's butterfly order (kpool_fp8_index.py
// _hadamard128): strides 1, 2 in the lane, then 4 .. 64 across lanes (xor 1 .. 16), then x fp32(128^-1/2)
template <int N>
__device__ __forceinline__ void hadamard128(float (&v)[N][4], int lane) {
#pragma unroll
  for (int j = 0; j < N; j++) {
    float a = v[j][0], b = v[j][1];
    v[j][0] = a + b;
    v[j][1] = a - b;
    a = v[j][2], b = v[j][3];
    v[j][2] = a + b;
    v[j][3] = a - b;
    a = v[j][0], b = v[j][2];
    v[j][0] = a + b;
    v[j][2] = a - b;
    a = v[j][1], b = v[j][3];
    v[j][1] = a + b;
    v[j][3] = a - b;
  }
#pragma unroll
  for (int m = 1; m < 32; m <<= 1) {
    bool const hi = (lane & m) != 0; // this lane holds b of each pair
#pragma unroll
    for (int j = 0; j < N; j++) {
#pragma unroll
      for (int i = 0; i < 4; i++) {
        float const o = __shfl_xor(v[j][i], m, 32);
        v[j][i] = hi ? o - v[j][i] : v[j][i] + o;
      }
    }
  }
#pragma unroll
  for (int j = 0; j < N; j++) {
#pragma unroll
    for (int i = 0; i < 4; i++) {
      v[j][i] *= 0.08838834764831845f;
    }
  }
}

// columns c .. c + 3 of token t of a gemm_tile node's K parts in slots ([P][T][NA] fp32, 0xFF-prefilled): polled, added
// in part order
template <int P, int NA>
__device__ __forceinline__ float4 slots_col4(float const *parts, int t, int c) {
  float4 z[P];
  bool all;
  do {
#pragma unroll
    for (int p = 0; p < P; p++) {
      z[p] = as_float4(ld16_at<AUX_DEV>(parts, (uint32_t)((((size_t)p * T + t) * NA + c) * 4)));
    }
    all = true;
#pragma unroll
    for (int p = 0; p < P; p++) {
      all = all && validf4(z[p]);
    }
    if (!all) {
      tail_pause();
    }
  } while (!all);
  float4 s = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
  for (int p = 0; p < P; p++) {
    s.x += z[p].x;
    s.y += z[p].y;
    s.z += z[p].z;
    s.w += z[p].w;
  }
  return s;
}

// a 4-B / 8-B device-scope store (write-through: other CUs read it after the task's count)
__device__ __forceinline__ void st4_dev(void *p, uint32_t v) {
  asm volatile("global_store_dword %0, %1, off sc1" ::"v"(p), "v"(v) : "memory");
}

} // namespace dsa

// a 4-B int at device scope (write-through)
__device__ __forceinline__ void st4_out_i(int *p, int v) {
  dsa::st4_dev(p, (uint32_t)v);
}

} // namespace static_mk
