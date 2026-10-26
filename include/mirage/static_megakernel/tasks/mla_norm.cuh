// tasks/mla_norm.cuh -- Kimi K3 MLA: the fused q_a / kv_a projection's epilogue
// (SGLang: the 2 RMSNorms after fused_a_gemm and set_mla_kv_buffer). Task t of
// grid (T, 1, 1) = token t:
//   z      = the projection's K parts added in part order (a gemm_tile node's
//   output in slots: [K parts][T][N] fp32,
//            0xFF-prefilled; N = Q + KV + ROPE = 1536 + 512 + 64), rounded to
//            bf16 (SGLang's fused_a_gemm writes bf16)
//   q_a    = bf16(z[:Q] * rsqrt(mean(z[:Q]^2) + eps_q) * w_q)            -> the
//   node's output [T][Q] bf16 (0xFF before each
//            launch: mla_qb polls it per 16 B)
//   k_nope = bf16(z[Q:Q+KV] * rsqrt(mean(.^2) + eps_kv) * w_kv); k_pe =
//   z[Q+KV:] the KV row [k_nope | k_pe] (KV + ROPE bf16) -> kv_new[t] (0xFF
//   before each launch, polled by mla_attention) and the
//            framework's KV cache at row loc[t] (SGLang's write location of
//            token t: out_cache_loc / the kernel-facing loc)
// The norm weights are read into shared memory before the poll (their latency
// hidden behind the projection). The parts re-armed (this task is their one
// reader).
#pragma once
#include "../core.cuh"
#include "mla_common.cuh"

namespace static_mk {

template <int NSPLIT, int N, int Q, int KV, int ROPE>
__device__ __forceinline__ void mla_norm_task(KernelLocals &L,
                                              float const *zpart,
                                              __nv_bfloat16 const *wq,
                                              __nv_bfloat16 const *wkv,
                                              float eps_q,
                                              float eps_kv,
                                              __nv_bfloat16 *q_a,
                                              __nv_bfloat16 *kv_new,
                                              __nv_bfloat16 *kv_cache,
                                              long long const *loc,
                                              int t) {
  static_assert(N == Q + KV + ROPE && N % 4 == 0 && Q % 8 == 0 && KV % 8 == 0 &&
                    ROPE % 8 == 0,
                "mla_norm: the row's parts");
  constexpr int NV4 = N / 4; // float4 vectors of the row
  int const tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  float *zs =
      reinterpret_cast<float *>(L.sm + RING_BYTES); // [N] the bf16-rounded row
  float *red = zs + N;          // [8 warps][2] sums of squares
  float *ws = red + 16;         // [Q + KV] the norm weights
  long long const row = loc[t]; // read before the poll (its latency hidden)
  __syncthreads(); // zs may be the SM's previous task's shared memory until
                   // here
  for (int c = tid; c < Q + KV; c += 256) {
    ws[c] = __bfloat162float(c < Q ? wq[c] : wkv[c - Q]);
  }
  float sq = 0.f, skv = 0.f;
  constexpr int PER = (NV4 + 255) / 256; // vectors per thread
  { // every vector of this thread, every part: polled together (one round trip
    // per round, not one per vector)
    float4 zp[PER][NSPLIT];
    bool got[PER] = {};
    bool all;
    do {
      all = true;
#pragma unroll
      for (int i = 0; i < PER; i++) {
        int const v = tid + 256 * i;
        if (v < NV4 && !got[i]) {
#pragma unroll
          for (int p = 0; p < NSPLIT; p++) {
            zp[i][p] = ldf4_relaxed(zpart + ((size_t)p * T + t) * N + 4 * v);
          }
        }
      }
#pragma unroll
      for (int i = 0; i < PER; i++) {
        int const v = tid + 256 * i;
        if (v < NV4 && !got[i]) {
          bool ok = true;
#pragma unroll
          for (int p = 0; p < NSPLIT; p++) {
            ok = ok && validf4(zp[i][p]);
          }
          got[i] = ok;
          all = all && ok;
        }
      }
    } while (!all);
#pragma unroll
    for (int i = 0; i < PER; i++) {
      int const v = tid + 256 * i;
      if (v >= NV4) {
        continue;
      }
      float z[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int p = 0; p < NSPLIT; p++) {
        z[0] += zp[i][p].x;
        z[1] += zp[i][p].y;
        z[2] += zp[i][p].z;
        z[3] += zp[i][p].w;
      }
#pragma unroll
      for (int j = 0; j < 4; j++) {
        float const bv = __bfloat162float(__float2bfloat16(z[j]));
        int const c = 4 * v + j;
        zs[c] = bv;
        if (c < Q) {
          sq += bv * bv;
        } else if (c < Q + KV) {
          skv += bv * bv;
        }
      }
      if constexpr (REARM) {
#pragma unroll
        for (int p = 0; p < NSPLIT; p++) {
          rearm16(zpart + ((size_t)p * T + t) * N + 4 * v);
        }
      }
    }
  }
#if defined(STATIC_TIMING_BUILD)
  if (tid == 0 && L.stage_stamps) {
    L.stage_stamps[0] = gtime();
  }
#endif
#pragma unroll
  for (int d = 16; d >= 1; d >>= 1) {
    sq += __shfl_xor_sync(0xffffffffu, sq, d);
    skv += __shfl_xor_sync(0xffffffffu, skv, d);
  }
  if (lane == 0) {
    red[2 * warp] = sq;
    red[2 * warp + 1] = skv;
  }
  __syncthreads();
  float tq = 0.f, tkv = 0.f;
#pragma unroll
  for (int w = 0; w < 8; w++) {
    tq += red[2 * w];
    tkv += red[2 * w + 1];
  }
  float const rq = rsqrtf(tq / (float)Q + eps_q),
              rkv = rsqrtf(tkv / (float)KV + eps_kv);
  // 16-B vectors: q_a (Q / 8), then the KV row (KV / 8 + ROPE / 8)
  for (int v = tid; v < (Q + KV + ROPE) / 8; v += 256) {
    int const c0 = 8 * v;
    float o[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
      int const c = c0 + j;
      float const x = zs[c];
      o[j] = c < Q ? x * rq * ws[c] : c < Q + KV ? x * rkv * ws[c] : x;
    }
    uint4 const pk = pack_bf16x8(make_float4(o[0], o[1], o[2], o[3]),
                                 make_float4(o[4], o[5], o[6], o[7]));
    if (c0 < Q) {
      *reinterpret_cast<uint4 *>(q_a + (size_t)t * Q + c0) = pk;
    } else {
      int const k = c0 - Q; // in the KV row
      *reinterpret_cast<uint4 *>(kv_cache + (size_t)row * (KV + ROPE) + k) = pk;
      *reinterpret_cast<uint4 *>(kv_new + (size_t)t * (KV + ROPE) + k) = pk;
    }
  }
  __syncthreads();
#if defined(STATIC_TIMING_BUILD)
  if (tid == 0 && L.stage_stamps) {
    L.stage_stamps[1] = gtime();
  }
#endif
}

// task t: Z = the gemm_tile node whose K parts it adds (grid y = the parts, N =
// its params[2]); params = [Q, KV, ROPE, eps_q as fp32 bits, eps_kv as fp32
// bits]; buffer slots = q_a_layernorm weight [Q] bf16, kv_a_layernorm weight
// [KV] bf16, kv_new [T][KV + ROPE] bf16 (0xFF each launch), the KV cache
// [rows][KV + ROPE] bf16, loc [T] int64
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class Z,
          class... IN>
__device__ __forceinline__ void run_mla_norm(Maps const &,
                                             G const &g,
                                             KernelLocals &L,
                                             StaticTask const &tk) {
  constexpr int Q = PARAMS::v[0], KV = PARAMS::v[1], ROPE = PARAMS::v[2];
  static_assert(
      Z::buf >= 0 && Z::v[0] == COMBINE_SLOTS && Z::v[2] == Q + KV + ROPE,
      "mla_norm: a gemm_tile node's K parts in slots, N = Q + KV + ROPE");
  static_assert(SELF::x == T && SELF::buf >= 0,
                "mla_norm: one task per token, its output");
  mla_norm_task<Z::y, Q + KV + ROPE, Q, KV, ROPE>(
      L,
      buf_at<float const>(g, Z::buf),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 1>()),
      __int_as_float(PARAMS::v[3]),
      __int_as_float(PARAMS::v[4]),
      buf_at<__nv_bfloat16>(g, SELF::buf),
      buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 2>()),
      buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 3>()),
      buf_at<long long const>(g, slot_at<BUF_SLOTS, 4>()),
      tk.x);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_mla_norm() {
  return RING_BYTES + (PARAMS::v[0] + PARAMS::v[1] + PARAMS::v[2]) * 4 + 64 +
         (PARAMS::v[0] + PARAMS::v[1]) * 4 + 16;
}

} // namespace static_mk
