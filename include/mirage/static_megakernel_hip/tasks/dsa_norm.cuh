// tasks/dsa_norm.cuh (HIP, gfx950) -- GLM-5.3-Flash DSA's 2 RMSNorms after the fused q_a / kv_a projection, as
// mla_norm.cuh (K3) with SGLang ROCm's fp8 latent cache (memory_pool.py DSATokenToKVPool, mla_buffer.py
// set_mla_kv_buffer_fp8_quant_kernel: the bf16 k_nope cast to e4m3, no scale, 512 B a row; no k_pe). Task (x, y) =
// token x, column range y of grid (T, NG, 1):
//   latent = bf16(the projection's K parts added) (a gemm_tile node's K parts in slots, [P][T][NA] fp32, polled)
//   q_c    = bf16(rmsnorm(latent[:QL]) w_q)       -> q_c [T][QL] bf16 (the node's buffer, 0xFF-prefilled: q_b and the
//            indexer's wq_b poll it)
//   k_nope = bf16(rmsnorm(latent[QL:QL + KV]) w_kv) -> e4m3 (round to nearest even) -> the cache row out_loc[x] (int64:
//            SGLang's out_cache_loc; a row 0 write is skipped as SGLang's: its pad slot), device scope
// Then one count on the node's counter.
#pragma once
#include "../core.cuh"

namespace static_mk {

// The norms' weights of this thread's output columns are loaded at the task's start (as mla_norm.cuh)
__device__ __forceinline__ void dsa_st4_dev(void *p, uint32_t v) {
  asm volatile("global_store_dword %0, %1, off sc1" ::"v"(p), "v"(v) : "memory");
}

// NG > 1 (grid (T, NG)): task (x, y) = token x, the y-th of NG column ranges
// (whole 4-column vectors): its range's K parts added, its partial sums of
// squares (q_a part, kv part) to the scratch [T][NG][2] fp32 (0xFF before each
// launch, data as the flag), the token's NG partials read in group order, then
// its range of q_a and of the KV row. One count per task.
template <int NPART, int NA, int QL, int KV, int RP, int NG>
__device__ __forceinline__ void dsa_norm_task(KernelLocals &L,
                                              float const *parts,
                                              __hip_bfloat16 const *wq,
                                              __hip_bfloat16 const *wkv,
                                              uint8_t *cache, // [rows][KV] e4m3
                                              long long const *out_loc,
                                              float eps_q,
                                              float eps_kv,
                                              __hip_bfloat16 *q_a,
                                              float *psum,    // NG > 1: [T][NG][2]
                                              uint32_t *cnt,
                                              int t,
                                              int y STAGE_STAMP_PARAM) {
  constexpr int NL = QL + KV + RP, D = KV + RP;
  static_assert(NL % 4 == 0 && NL <= NA && QL % 4 == 0 && KV % 4 == 0 && RP % 4 == 0,
                "dsa_norm: whole 4-column pieces");
  static_assert(RP == 0, "dsa_norm: no k_pe (GLM-5.3-Flash: NoPE)");
  constexpr int NV = NL / 4, NVG = (NV + NG - 1) / NG, PER = (NVG + 255) / 256;
  int const tid = task_tid_all(), v_lo = NVG * y, v_hi = min(NV, v_lo + NVG);
  float *const lat = reinterpret_cast<float *>(L.sm); // [NL] the bf16 latent row (this range's part)
  float *const red = lat + NL;                         // [2][4 waves]
  long long const loc = out_loc[t];
  uint8_t *const crow = cache + (size_t)loc * D;
  __syncthreads();
  uint2 wpre[PER]; // (k_pe columns: unused)
#pragma unroll
  for (int i = 0; i < PER; i++) {
    int const c = 4 * min(v_lo + tid + 256 * i, v_hi - 1);
    wpre[i] = c < QL ? *reinterpret_cast<uint2 const *>(wq + c)
                     : *reinterpret_cast<uint2 const *>(wkv + min(c - QL, KV - 4));
  }
  float sq = 0.f, sk = 0.f; // this thread's sums of squares (q_a part, kv part)
#pragma unroll
  for (int i = 0; i < PER; i++) {
    int const v = v_lo + tid + 256 * i;
    if (v < v_hi) {
      float4 zp[NPART];
      bool all;
      do {
#pragma unroll
        for (int p = 0; p < NPART; p++) {
          zp[p] = as_float4(ld16_at<AUX_DEV>(parts, (uint32_t)((((size_t)p * T + t) * NA + 4 * v) * 4))); // (all in flight)
        }
        all = true;
#pragma unroll
        for (int p = 0; p < NPART; p++) {
          all = all && validf4(zp[p]);
        }
        if (!all) {
          tail_pause();
        }
      } while (!all);
      float z[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
      for (int p = 0; p < NPART; p++) {
        z[0] += zp[p].x;
        z[1] += zp[p].y;
        z[2] += zp[p].z;
        z[3] += zp[p].w;
      }
#pragma unroll
      for (int e = 0; e < 4; e++) {
        z[e] = round_bf16(z[e]);
        lat[4 * v + e] = z[e];
        int const c = 4 * v + e;
        if (c < QL) {
          sq = fmaf(z[e], z[e], sq);
        } else if (c < QL + KV) {
          sk = fmaf(z[e], z[e], sk);
        }
      }
    }
  }
  if (tid == 0) { // (timing build: stage stamp 0 = thread 0's K parts landed)
    STAGE_STAMP_AT(0)
  }
  sq = wave_sum64(sq);
  sk = wave_sum64(sk);
  if ((tid & 63) == 0) {
    red[tid >> 6] = sq;
    red[4 + (tid >> 6)] = sk;
  }
  __syncthreads();
  float tq = (red[0] + red[1]) + (red[2] + red[3]), tk = (red[4] + red[5]) + (red[6] + red[7]);
  if constexpr (NG > 1) { // the token's NG partials (this one's stored first), in group order
    float *const pt = psum + (size_t)t * NG * 2;
    if (tid == 0) {
      dsa_st4_dev(pt + 2 * y, __float_as_uint(tq));
      dsa_st4_dev(pt + 2 * y + 1, __float_as_uint(tk));
    }
    if (tid < 2 * NG) {
      float v = __uint_as_float(__builtin_amdgcn_raw_buffer_load_b32(buf_rsrc(pt), (uint32_t)(tid * 4), 0, AUX_DEV));
      while (__float_as_uint(v) == 0xFFFFFFFFu) {
        tail_pause();
        v = __uint_as_float(__builtin_amdgcn_raw_buffer_load_b32(buf_rsrc(pt), (uint32_t)(tid * 4), 0, AUX_DEV));
      }
      red[8 + tid] = v;
    }
    __syncthreads();
    tq = 0.f;
    tk = 0.f;
#pragma unroll
    for (int gq = 0; gq < NG; gq++) {
      tq += red[8 + 2 * gq];
      tk += red[8 + 2 * gq + 1];
    }
  }
  float const rq = rsqrtf(tq / QL + eps_q);
  float const rk = rsqrtf(tk / KV + eps_kv);
  // this range's outputs, 4 columns (8 B) per thread: q_a, k_nope (normed), k_pe
#pragma unroll
  for (int i = 0; i < PER; i++) {
    int const v = v_lo + tid + 256 * i;
    if (v >= v_hi) {
      break;
    }
    int const c = 4 * v;
    float o4[4];
    if (c < QL) {
      uint2 const w2 = wpre[i];
      o4[0] = lat[c] * rq * bf16_lo(w2.x);
      o4[1] = lat[c + 1] * rq * bf16_hi(w2.x);
      o4[2] = lat[c + 2] * rq * bf16_lo(w2.y);
      o4[3] = lat[c + 3] * rq * bf16_hi(w2.y);
      v2u const w = {pack_bf16x2(o4[0], o4[1]), pack_bf16x2(o4[2], o4[3])};
      asm volatile("global_store_dwordx2 %0, %1, off sc1" ::"v"(q_a + (size_t)t * QL + c), "v"(w) : "memory");
    } else {
      int const k = c - QL;
      if (k < KV) {
        uint2 const w2 = wpre[i];
        o4[0] = lat[c] * rk * bf16_lo(w2.x);
        o4[1] = lat[c + 1] * rk * bf16_hi(w2.x);
        o4[2] = lat[c + 2] * rk * bf16_lo(w2.y);
        o4[3] = lat[c + 3] * rk * bf16_hi(w2.y);
      } else {
#pragma unroll
        for (int e = 0; e < 4; e++) {
          o4[e] = lat[c + e];
        }
      }
      // bf16 (SGLang's kv_a_layernorm output), then e4m3 (v_cvt_pk_fp8_f32: OCP e4m3, round to nearest even)
      int w = __builtin_amdgcn_cvt_pk_fp8_f32(round_bf16(o4[0]), round_bf16(o4[1]), 0, false);
      w = __builtin_amdgcn_cvt_pk_fp8_f32(round_bf16(o4[2]), round_bf16(o4[3]), w, true);
      if (loc != 0) {
        dsa_st4_dev(crow + k, (uint32_t)w);
      }
    }
  }
  if (tid == 0) { // (stage stamp 1 = the outputs' stores issued)
    STAGE_STAMP_AT(1)
  }
  task_post(cnt, 1);
}

// task x = token x; Z = the fused projection's gemm_tile (K parts in slots;
// NA = its params[2]); params = [QL, KV, RP, eps_q bits, eps_kv bits]; buffer
// slots = w_q norm, w_kv norm, the KV cache, out_loc
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class Z,
          class... IN>
__device__ __forceinline__ void run_dsa_norm(Maps const &,
                                             G const &g,
                                             KernelLocals &L,
                                             StaticTask const &tk) {
  static_assert(Z::buf >= 0 && Z::v[0] == COMBINE_SLOTS,
                "dsa_norm: a gemm_tile node's K parts in slots");
  static_assert(SELF::buf >= 0 && SELF::counter >= 0 && SELF::x == T,
                "dsa_norm: one task per token (and column range), its output and a counter");
  float *psum = nullptr;
  if constexpr (SELF::y > 1) {
    psum = buf_at<float>(g, slot_at<BUF_SLOTS, 4>());
  }
  dsa_norm_task<Z::y, Z::v[2], PARAMS::v[0], PARAMS::v[1], PARAMS::v[2], SELF::y>(
      L,
      buf_at<float const>(g, Z::buf),
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<uint8_t>(g, slot_at<BUF_SLOTS, 2>()),
      buf_at<long long const>(g, slot_at<BUF_SLOTS, 3>()),
      __int_as_float(PARAMS::v[3]),
      __int_as_float(PARAMS::v[4]),
      buf_at<__hip_bfloat16>(g, SELF::buf),
      psum,
      cnt_of(g) + SELF::counter,
      tk.x,
      tk.y STAGE_STAMP_ARG(L.stage_stamps));
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_dsa_norm() {
  return (PARAMS::v[0] + PARAMS::v[1] + PARAMS::v[2] + 8 + 2 * SELF::y) * 4;
}

} // namespace static_mk
