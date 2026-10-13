// tasks/hc_post.cuh -- mHC's post step after a TP sublayer (vLLM's
// mhc_post_torch): task (q, t) of grid (Q, T, 1) = token t, H / Q columns
// [q C, q C + C): add the GPUs' bf16 partial sums of the sublayer output (an
// allreduce_send node's buffer, core.cuh allreduce_land_bf16: fp32, GPU
// order) -> out (bf16), then for every stream j
//   R'[t][j][c] = bf16(sum_i comb[i][j] R[t][i][c] + post[j] out[c])
// with post and comb finished here (warp 0, hc_comb_post, while the sums are
// on their way) from the z hc_pre leaves after x (HC_COEF floats per token).
#pragma once
#include "../core.cuh"
#include "hc_pre.cuh"

namespace static_mk {

template <size_t SOFF, int H, int HC, int Q, int ITERS> // SOFF: the sent
                                                        // sums' offset in the
                                                        // exchange region
__device__ __forceinline__ void hc_post_task(G const &g,
                                             float const *coef,
                                             float hc_eps,
                                             float post_mult,
                                             uint32_t const *pre_cnt,
                                             uint32_t pre_tasks,
                                             __nv_bfloat16 const *res,
                                             __nv_bfloat16 *rout,
                                             int q,
                                             int t STAGE_STAMP_PARAM) {
  constexpr size_t S_RANK = (size_t)T * H * 2; // one GPU's slot
  constexpr int COLS = H / Q;
  static_assert(COLS * Q == H && COLS % 8 == 0 && COLS / 8 <= 256,
                "hc_post: whole 8-column vectors, one per thread");
  int const v = threadIdx.x;
  int const c = q * COLS + 8 * v;
  uint4 ru[HC]; // R[t], columns c .. c + 7: loaded before the waits
  if (v < COLS / 8) {
#pragma unroll
    for (int i = 0; i < HC; i++) {
      ru[i] =
          *reinterpret_cast<uint4 const *>(res + ((size_t)t * HC + i) * H + c);
    }
  }
  __shared__ float coef_s[HC_COEF];
  if (threadIdx.x == 0) {
    cnt_wait(pre_cnt, pre_tasks); // acquire: hc_pre's coefficients
    STAGE_STAMP_AT(0)
  }
  __syncthreads();
  // post and comb from hc_pre's z (warp 0), while the sums are on their way
  if (threadIdx.x < 32) {
    int const ci = hc_coef_index<HC>(threadIdx.x);
    coef_s[ci] = hc_comb_post<HC, ITERS>(
        coef[(size_t)t * HC_COEF + ci], threadIdx.x, hc_eps, post_mult);
  }
  __syncthreads();
  if (v < COLS / 8) {
    float post[HC], comb[HC][HC];
#pragma unroll
    for (int j = 0; j < HC; j++) {
      post[j] = coef_s[HC + j];
#pragma unroll
      for (int i = 0; i < HC; i++) {
        comb[i][j] = coef_s[2 * HC + i * HC + j];
      }
    }
    unsigned char const *row =
        g.rv + SOFF + rg_set() + (size_t)t * H * 2 + (size_t)q * COLS * 2;
    float s[8];
    allreduce_land_bf16<S_RANK>(g, row, v, s);
    if (v == 0) {
      STAGE_STAMP_AT(1)
    } // timing build: the first 8 columns of every GPU landed
#pragma unroll
    for (int k = 0; k < 8; k++) {
      s[k] = __bfloat162float(__float2bfloat16(s[k])); // the all-reduced bf16
    }
#pragma unroll
    for (int j = 0; j < HC; j++) {
      uint4 ou;
      __nv_bfloat16 *ob = reinterpret_cast<__nv_bfloat16 *>(&ou);
#pragma unroll
      for (int k = 0; k < 8; k++) {
        float a = 0.f;
#pragma unroll
        for (int i = 0; i < HC; i++) {
          a += comb[i][j] *
               __bfloat162float(
                   reinterpret_cast<__nv_bfloat16 const *>(&ru[i])[k]);
        }
        ob[k] = __float2bfloat16(a + post[j] * s[k]);
      }
      *reinterpret_cast<uint4 *>(rout + ((size_t)t * HC + j) * H + c) = ou;
    }
    if constexpr (REARM) {
      allreduce_rearm<S_RANK>(g, row, v);
    }
  }
  __syncthreads();
}

// hc_post task (part x, token y). SEND: the allreduce_send node whose output it
// adds (params {H, 1}); HCPRE: the hc_pre node of this sublayer (its buffer
// holds x, then the coefficients; its counter counts its T tasks; its params:
// hc eps, the post multiplier, the Sinkhorn iterations); PARAMS {H, HC};
// BUF_SLOTS {the residual streams R, the output R' ([T][HC H] bf16 graph
// tensors)}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class SEND,
          class RES,
          class HCPRE,
          class... IN>
__device__ __forceinline__ void run_hc_post(Maps const &,
                                            G const &g,
                                            KernelLocals &L,
                                            StaticTask const &tk) {
  constexpr int H = PARAMS::v[0], HC = PARAMS::v[1];
  static_assert(SEND::buf >= 0 && SEND::v[0] == H && SEND::v[1] == 1,
                "hc_post: rows of H sent by an allreduce_send node");
  static_assert(
      exchange_bytes[SEND::buf] == (size_t)GPUS * T * H * 2,
      "hc_post: the sent sums, [GPUS][T][H] bf16 in the exchange region");
  static_assert(HCPRE::buf >= 0 && HCPRE::counter >= 0,
                "hc_post: hc_pre's buffer (x, then the coefficients)");
  static_assert(SELF::y == T, "hc_post: grid (parts, T tokens)");
  hc_post_task<exchange_offset[SEND::buf], H, HC, SELF::x, HCPRE::v[5]>(
      g,
      reinterpret_cast<float const *>(
          buf_at<__nv_bfloat16 const>(g, HCPRE::buf) + (size_t)T * H),
      __int_as_float(HCPRE::v[2]),
      __int_as_float(HCPRE::v[4]),
      g.cnt + HCPRE::counter,
      HCPRE::x * HCPRE::y * HCPRE::z,
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 1>()),
      tk.x,
      tk.y STAGE_STAMP_ARG(L.stage_stamps));
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_hc_post() {
  return 0;
}

} // namespace static_mk
