// tasks/mla_absorb.cuh (HIP, gfx950) -- K3 MLA's query absorption (math:
// mpk-kernels tests/mla_reference.py core(); B300 design: mirage_kda
// tasks/mla_absorb.cuh): task (x, y) of grid (heads, KV / 128, 1) = head x, the
// latent rows [128 y, 128 y + 128):
//   q_abs[t][l] = bf16(sum_n q_nope[t][n] w_kc[n][l])   (q_nope: q_b's output of
//            this head, [T][heads x (NOPE + RP)] bf16, 0xFF-prefilled, polled;
//            w_kc given as [heads][KV][NOPE] bf16: SGLang's memory layout)
//   -> the query [T][heads][KV + RP] bf16 (the node's buffer, 0xFF-prefilled:
//            mla_attn polls it); task y = 0 also copies q_pe into its last RP
// By MFMA 16x16x32 bf16: wave w takes rows 32 w .. + 31 of the task's 128 (2
// tiles), K = n (NOPE / 32 steps); its w_kc fragments are loaded into registers
// as the task starts (no input needed), the tokens' q_nope is polled after.
#pragma once
#include "../core.cuh"
#include "mla_norm.cuh" // mla_st4_dev

namespace static_mk {

template <int HEADS, int NOPE, int RP, int KV>
__device__ __forceinline__ void mla_absorb_task(KernelLocals &L,
                                                __hip_bfloat16 const *q,     // [T][HEADS x (NOPE + RP)]
                                                __hip_bfloat16 const *w_kc,  // [HEADS][KV][NOPE]
                                                __hip_bfloat16 *query,       // [T][HEADS][KV + RP]
                                                int h,
                                                int y) {
  static_assert(NOPE == 128 && KV % 128 == 0 && T <= 64 && RP % 8 == 0, "mla_absorb: K3's shapes (thread maps below)");
  constexpr int NC = (T + 15) / 16; // the tokens as the MFMA's columns: tiles of 16
  constexpr int QW = HEADS * (NOPE + RP), D = KV + RP, S = NOPE / 32;
  int const tid = task_tid_all(), lane = tid & 63, wave = tid >> 6, tok = lane % 16, grp = lane / 16;
  __hip_bfloat16 *const qn = reinterpret_cast<__hip_bfloat16 *>(L.sm); // [T][NOPE]
  // 1. this wave's w_kc^T fragments: rows l = 128 y + 32 wave + 16 i + l % 16,
  // n = 32 s + 8 (l / 16) .. + 7
  v4f a[2][S];
#pragma unroll
  for (int i = 0; i < 2; i++) {
    int const l = 128 * y + 32 * wave + 16 * i + tok;
#pragma unroll
    for (int s = 0; s < S; s++) {
      a[i][s] = *reinterpret_cast<v4f const *>(w_kc + ((size_t)h * KV + l) * NOPE + 32 * s + 8 * grp);
    }
  }
  __syncthreads(); // the LDS may still be read by the previous task
  // 2. q_nope of this head (polled; T x NOPE / 8 vectors), and (y = 0) q_pe
  // copied to the query (polled the same way)
  for (int v = tid; v < T * (NOPE + RP) / 8; v += 256) {
    if (v < T * NOPE / 8) {
      int const t = v / (NOPE / 8), c = 8 * (v % (NOPE / 8));
      *reinterpret_cast<uint4 *>(qn + t * NOPE + c) = poll16(q + (size_t)t * QW + h * (NOPE + RP) + c);
    } else if (y == 0) {
      int const i = v - T * NOPE / 8, t = i / (RP / 8), c = 8 * (i % (RP / 8));
      uint4 const v4 = poll16(q + (size_t)t * QW + h * (NOPE + RP) + NOPE + c);
      __hip_bfloat16 *const dst = query + ((size_t)t * HEADS + h) * D + KV + c;
      st16_dev(dst, v4);
    }
  }
  __syncthreads();
  // 3. the MFMAs (B: the tokens' q_nope; D: lane -> token l % 16, register r ->
  // row 4 (l / 16) + r)
  v4f acc[NC][2];
#pragma unroll
  for (int c = 0; c < NC; c++) {
    acc[c][0] = acc[c][1] = v4f{0.f, 0.f, 0.f, 0.f};
  }
#pragma unroll
  for (int s = 0; s < S; s++) {
#pragma unroll
    for (int c = 0; c < NC; c++) {
      v4f b = {0.f, 0.f, 0.f, 0.f};
      if (16 * c + tok < T) {
        b = *reinterpret_cast<v4f const *>(qn + (16 * c + tok) * NOPE + 32 * s + 8 * grp);
      }
#pragma unroll
      for (int i = 0; i < 2; i++) {
        acc[c][i] = __builtin_amdgcn_mfma_f32_16x16x32_bf16(__builtin_bit_cast(v8bf, a[i][s]), __builtin_bit_cast(v8bf, b),
                                                            acc[c][i], 0, 0, 0);
      }
    }
  }
  __syncthreads(); // every wave's reads of qn done: the task's [T][128] outputs staged there (bf16, NOPE == 128)
#pragma unroll
  for (int c = 0; c < NC; c++) {
    int const t = 16 * c + tok;
    if (t < T) {
#pragma unroll
      for (int i = 0; i < 2; i++) {
        int const r0 = 32 * wave + 16 * i + 4 * grp;
        *reinterpret_cast<uint2 *>(qn + t * NOPE + r0) =
            make_uint2(pack_bf16x2(acc[c][i][0], acc[c][i][1]), pack_bf16x2(acc[c][i][2], acc[c][i][3]));
      }
    }
  }
  __syncthreads();
  for (int v = tid; v < T * 128 / 8; v += 256) { // each token's 256 B of these latent rows in 16-B stores
    int const t = v / 16, c = 8 * (v % 16);
    st16_dev(query + ((size_t)t * HEADS + h) * D + 128 * y + c, *reinterpret_cast<uint4 const *>(qn + t * NOPE + c));
  }
  __syncthreads();
}

// task (x, y) = head x, latent rows [128 y, + 128); Q = q_b (a local gemm_tile:
// its buffer [T][heads x (NOPE + RP)]); params = [HEADS, NOPE, RP, KV]; buffer
// slots = w_kc ([heads][KV][NOPE])
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class Q,
          class... IN>
__device__ __forceinline__ void run_mla_absorb(Maps const &,
                                               G const &g,
                                               KernelLocals &L,
                                               StaticTask const &tk) {
  static_assert(Q::buf >= 0 && Q::v[0] == COMBINE_LOCAL, "mla_absorb: q_b, a local gemm_tile's output");
  static_assert(SELF::buf >= 0 && SELF::x == PARAMS::v[0] && SELF::y * 128 == PARAMS::v[3],
                "mla_absorb: one task per head and 128 latent rows, its output");
  mla_absorb_task<PARAMS::v[0], PARAMS::v[1], PARAMS::v[2], PARAMS::v[3]>(
      L,
      buf_at<__hip_bfloat16 const>(g, Q::buf),
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<__hip_bfloat16>(g, SELF::buf),
      tk.x,
      tk.y);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_mla_absorb() {
  return T * PARAMS::v[1] * 2;
}

} // namespace static_mk
