// tasks/glm53_index_k.cuh (HIP, gfx950) -- GLM-5.3 DSA indexer, the k side and the dense row list (SGLang v0.5.22 ROCm: the gfx950
// fused indexer's qk_rope_hadamard_quant k side, kernels/jit/csrc/dsa_gfx950/qk_rope_hadamard_quant.cuh:161-217, and the k-only
// path's row list for seq_len <= K, dsa_indexer.py:968-1003, 1775-1790; amd/notes/sglang_v0522_glm53_layer_contract.md D3, D6.2).
// Task x of grid (T, 1, 1) = token x:
//   k     = bf16(wk's K parts added) (a gemm_tile node's K parts in slots, [P][T][NA] fp32, polled; columns 0 .. 127)
//   k     = bf16(LayerNorm(k) fp32 (eps), fp32 weight / bias); dims 0 .. 63: interleaved rope (pairs (2i, 2i + 1), bf16 cos / sin
//           [positions][32] at pos[x]) -> bf16; Hadamard-128 x 128^-0.5 -> bf16
//   scale = 2^ceil(log2(max(amax, 1e-4) / 448)); e4m3(k / scale) into SGLang's index-K page of slot out_loc[x] (page = slot / 64:
//           bytes [0, 8192) preshuffled 16 x 16 tiles (token x dim), [8192, 8448) one fp32 scale per token); a negative slot skips
//   rows  = the attention's list [T][K] int32: seq_len <= K: req_to_token[req_idx[x]][p] for p < seq_len, then -1 (SGLang's dense
//           variant); the node's buffer, and a copy into a graph tensor (the shared layers' list in later launches)
// (seq_len > K needs the scored top-K: not here - the caller keeps seq_len <= K.) Wave 0: the k side; waves 1 .. 3: the list.
// One count per task.
#pragma once
#include "../core.cuh"

namespace static_mk {

template <int NPART, int NA, int K>
__device__ __forceinline__ void glm53_index_k_task(float const *parts,
                                                   float const *nw,  // LayerNorm weight [128] fp32
                                                   float const *nb,  // bias [128] fp32
                                                   long long const *pos,
                                                   __hip_bfloat16 const *cos_t, // [positions][32]
                                                   __hip_bfloat16 const *sin_t,
                                                   long long const *out_loc,
                                                   uint8_t *kcache, // [pages][64 (128 + 4)]
                                                   int const *table, int tw, long long const *req_idx, int const *seq_lens,
                                                   int *rows, int *rows_g, // [T][K]
                                                   uint32_t *cnt, int x, float eps) {
#pragma clang fp contract(off)
  constexpr int HD = 128, HALF = 64, PAGE = 64;
  static_assert(NA >= HD && NA % 4 == 0 && K % 256 == 0, "glm53_index_k: wk's 128 columns, a list of whole 256-entry rounds");
  int const tid = STATIC_TIDX, wave = tid >> 6, t = tid & 63;
  if (wave == 0) {
    long long const slot = out_loc[x], p = pos[x];
    // the two columns t, t + 64 of wk's output: its K parts added (polled), bf16
    float z[2] = {0.f, 0.f};
#pragma unroll
    for (int h = 0; h < 2; h++) {
      float zp[NPART];
      bool all;
      do {
        all = true;
#pragma unroll
        for (int q = 0; q < NPART; q++) {
          zp[q] = __uint_as_float(__builtin_amdgcn_raw_buffer_load_b32(
              buf_rsrc(parts), (uint32_t)((((size_t)q * T + x) * NA + t + h * HALF) * 4), 0, AUX_DEV));
          all = all && __float_as_uint(zp[q]) != 0xFFFFFFFFu;
        }
        if (!all) {
          tail_pause();
        }
      } while (!all);
#pragma unroll
      for (int q = 0; q < NPART; q++) {
        z[h] += zp[q];
      }
      z[h] = round_bf16(z[h]);
    }
    float const mean = (wave_sum64(z[1]) + wave_sum64(z[0])) / (float)HD;
    float const clo = z[0] - mean, chi = z[1] - mean;
    float const ss = wave_sum64(chi * chi) + wave_sum64(clo * clo);
    float const inv = rsqrtf(ss / (float)HD + eps);
    float klo = round_bf16(clo * inv * nw[t] + nb[t]);
    float khi = round_bf16(chi * inv * nw[t + HALF] + nb[t + HALF]);
    { // rope on dims 0 .. 63 (lane t: dim t; its pair t ^ 1)
      float const pv = __shfl_xor(klo, 1);
      float const cs = __bfloat162float(cos_t[(size_t)p * 32 + t / 2]), sn = __bfloat162float(sin_t[(size_t)p * 32 + t / 2]);
      klo = round_bf16((t & 1) == 0 ? klo * cs - pv * sn : klo * cs + pv * sn);
    }
#pragma unroll
    for (int s = 0; s < 6; s++) { // Hadamard-128: strides 1 .. 32 within the wave, then 64 across lo / hi
      int const d = 1 << s;
      float const plo = __shfl_xor(klo, d), phi = __shfl_xor(khi, d);
      bool const a = (t & d) == 0;
      klo = a ? klo + plo : plo - klo;
      khi = a ? khi + phi : phi - khi;
    }
    float const hs = 0.08838834764831845f; // 128^-0.5
    float const nlo = klo + khi, nhi = klo - khi;
    klo = round_bf16(nlo * hs);
    khi = round_bf16(nhi * hs);
    float const amax = fmaxf(wave_max64(fabsf(khi)), wave_max64(fabsf(klo)));
    float const ksc = exp2f(ceilf(log2f(fmaxf(amax, 1e-4f) / 448.f)));
    float const kin = 1.f / ksc;
    if (slot >= 0) {
      long long const blk = slot / PAGE, off = slot % PAGE;
      uint8_t *const page = kcache + (size_t)blk * PAGE * (HD + 4);
      if (t == 0) { // (plain stores: index-K is read by later launches only)
        *reinterpret_cast<float *>(page + PAGE * HD + off * 4) = ksc;
      }
#pragma unroll
      for (int h = 0; h < 2; h++) {
        int const dim = t + h * HALF;
        int const w = __builtin_amdgcn_cvt_pk_fp8_f32((h == 0 ? klo : khi) * kin, 0.f, 0, false);
        size_t const o = (size_t)(off / 16) * (16 * HD) + (dim / 16) * 256 + (off % 16) * 16 + dim % 16;
        page[o] = (uint8_t)(w & 0xFF);
      }
    }
  } else { // the dense list: every position's slot in order, then -1
    int const len = min(max(seq_lens[x], 0), min(K, tw));
    int const *const rt = table + (size_t)req_idx[x] * (size_t)tw;
    for (int i = tid - 64; i < K; i += 192) {
      int const r = i < len ? rt[i] : -1;
      __hip_atomic_store(rows + (size_t)x * K + i, r, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      rows_g[(size_t)x * K + i] = r;
    }
  }
  __syncthreads();
  task_post(cnt, 1);
}

// task x = token x; Z = wk's gemm_tile (K parts in slots; NA = its params[2]); params = [K, eps bits, table row stride];
// buffer slots = LayerNorm weight, bias (fp32 [128]), positions, cos, sin, out_loc, the index-K cache, req_to_token, req_idx,
// seq_lens, the rows' graph copy; its buffer: the rows [T][K] int32; its counter: the count of its tasks
template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class Z, class... IN>
__device__ __forceinline__ void run_glm53_index_k(Maps const &, G const &g, KernelLocals &L, StaticTask const &tk) {
  static_assert(Z::buf >= 0 && Z::v[0] == COMBINE_SLOTS, "glm53_index_k: wk's gemm_tile K parts in slots");
  static_assert(SELF::buf >= 0 && SELF::counter >= 0 && SELF::x == T, "glm53_index_k: a task per token, its rows, a counter");
  glm53_index_k_task<Z::y, Z::v[2], PARAMS::v[0]>(
      buf_at<float const>(g, Z::buf), buf_at<float const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 1>()), buf_at<long long const>(g, slot_at<BUF_SLOTS, 2>()),
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 3>()), buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 4>()),
      buf_at<long long const>(g, slot_at<BUF_SLOTS, 5>()), buf_at<uint8_t>(g, slot_at<BUF_SLOTS, 6>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 7>()), PARAMS::v[2], buf_at<long long const>(g, slot_at<BUF_SLOTS, 8>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 9>()), buf_at<int>(g, SELF::buf), buf_at<int>(g, slot_at<BUF_SLOTS, 10>()),
      cnt_of(g) + SELF::counter, tk.x, __int_as_float(PARAMS::v[1]));
}

template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class... IN>
constexpr int smem_glm53_index_k() {
  return 0;
}

} // namespace static_mk
