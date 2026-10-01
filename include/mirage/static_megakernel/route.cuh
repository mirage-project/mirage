// route.cuh -- the routing of one token t: add the router's NSPLIT partial sums of row t (lpart, in a fixed order so that every
// GPU routes the same way), sigmoid, + score correction bias, top 16 of the NE experts, renormalise the 16 sigmoid scores ->
// 16 (expert, weight) pairs, written to pairs64 as expert << 32 | weight bits (0xFF-prefilled: the expert queue polls them).
// Two passes: pass 0 runs the same code on bias-only scores without loading anything (this code runs once per SM, cold; the dry
// pass loads its instructions), pass 1 is the real one. 256 threads.
#pragma once
#include "runtime.cuh"
#include "moe_types.cuh"

namespace static_mk {

template <int NSPLIT>
__device__ __forceinline__ void route_task(G const &g, int t, char *sm) {
  int const warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  __syncthreads();   // the scratch below is in the ring area: the previous tile's epilogue must have read every landed stage
  // scratch: sc_s [NE] sigmoid scores, [NE] not used, key_s [NE] + cand_s [128] 64-bit keys, sel_s [16], wsel_s [16]
  float *sc_s = reinterpret_cast<float *>(sm + OFF_GLUE);
  uint32_t *hist = reinterpret_cast<uint32_t *>(sc_s + 2 * NE);
  int *sel_s = reinterpret_cast<int *>(hist + 2 * (NE + 128));
  float *wsel_s = reinterpret_cast<float *>(sel_s + 16);
  unsigned long long *key_s = reinterpret_cast<unsigned long long *>(hist);   // key = orderable (score + bias) << 32 | (~expert id)
  unsigned long long *cand_s = key_s + NE;
  for (int pass = 0; pass < 2; pass++) {
    bool const real = pass == 1;
    // scores: thread q < NE / 4 owns experts 4q .. 4q + 3; the NSPLIT partial sums are loaded together (float4, independent loads)
    if (threadIdx.x < NE / 4) {
      float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
      float4 lp[NSPLIT];
#pragma unroll
      for (int p_ = 0; p_ < NSPLIT; p_++) lp[p_] = make_float4(0.f, 0.f, 0.f, 0.f);   // dry pass: scores from the bias only
      if (real) {   // re-load all parts until every one has landed (0xFF prefill)
        bool all;
        do {
#pragma unroll
          for (int p_ = 0; p_ < NSPLIT; p_++) lp[p_] = ldf4_relaxed(g.lpart + ((size_t)p_ * T + t) * NE + threadIdx.x * 4);
          all = true;
#pragma unroll
          for (int p_ = 0; p_ < NSPLIT; p_++) all = all && validf4(lp[p_]);
        } while (!all);
      }
#pragma unroll
      for (int p_ = 0; p_ < NSPLIT; p_++) { acc.x += lp[p_].x; acc.y += lp[p_].y; acc.z += lp[p_].z; acc.w += lp[p_].w; }
      float4 const b4 = *reinterpret_cast<float4 const *>(g.gate_bias + threadIdx.x * 4);
      float const lg4[4] = {acc.x, acc.y, acc.z, acc.w}, bs4[4] = {b4.x, b4.y, b4.z, b4.w};
#pragma unroll
      for (int k = 0; k < 4; k++) {
        int const e = threadIdx.x * 4 + k;
        float const sg = 1.0f / (1.0f + __expf(-lg4[k]));
        sc_s[e] = sg;
        uint32_t u = __float_as_uint(sg + bs4[k]);
        u = (u & 0x80000000u) ? ~u : (u | 0x80000000u);   // float -> unsigned with the same order
        key_s[e] = ((unsigned long long)u << 32) | (unsigned long long)(0xFFFFFFFFu - (uint32_t)e);   // ties: lower expert id wins
      }
    }
    __syncthreads();
    // stage 1: warp w ranks its 112 keys among themselves; the 16 with rank < 16 go to cand_s[w * 16 + rank]
    {
      unsigned long long mine[4]; int rk[4];
#pragma unroll
      for (int q = 0; q < 4; q++) { int const i = lane + 32 * q; mine[q] = (i < 112) ? key_s[warp * 112 + i] : 0ull; rk[q] = 0; }
#pragma unroll 16
      for (int j = 0; j < 112; j++) {
        unsigned long long const o = key_s[warp * 112 + j];
#pragma unroll
        for (int q = 0; q < 4; q++) rk[q] += (o > mine[q]);
      }
#pragma unroll
      for (int q = 0; q < 4; q++) if (lane + 32 * q < 112 && rk[q] < 16) cand_s[warp * 16 + rk[q]] = mine[q];
    }
    __syncthreads();
    // stage 2: 128 threads rank the 128 candidates; rank < 16 -> sel_s[rank] (keys are unique, so ranks are unique)
    if (threadIdx.x < 128) {
      unsigned long long const mine = cand_s[threadIdx.x];
      int rk = 0;
#pragma unroll 16
      for (int j = 0; j < 128; j++) rk += (cand_s[j] > mine);
      if (rk < 16) sel_s[rk] = (int)(0xFFFFFFFFu - (uint32_t)(mine & 0xFFFFFFFFull));
    }
    __syncthreads();
    if (threadIdx.x == 0) { for (int j = 0; j < 16; j++) if (sel_s[j] < 0 || sel_s[j] >= NE) sel_s[j] = 0; }   // never triggers
    __syncthreads();
    if (threadIdx.x < 16) wsel_s[threadIdx.x] = sc_s[sel_s[threadIdx.x]];
    __syncthreads();
    if (threadIdx.x < 16) {
      float sum = 0.f;
      for (int k = 0; k < 16; k++) sum += wsel_s[k];
      if (real) {
        float const w = wsel_s[threadIdx.x] / sum;
        g.pairs64[t * 16 + threadIdx.x] = ((unsigned long long)(unsigned)sel_s[threadIdx.x] << 32) | (unsigned long long)__float_as_uint(w);
      }
    }
    __syncthreads();
  }
}

}  // namespace static_mk
