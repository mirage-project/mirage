// tasks/hc_post.cuh -- mHC's post step after a TP sublayer (SGLang's
// mhc_post_torch): task (q, t) of grid (Q, T, 1) = token t, H / Q columns
// [q C, q C + C): add the GPUs' bf16 partial sums of the sublayer output (an
// allreduce_send node's buffer, core.cuh allreduce_land_bf16: fp32, GPU
// order) -> out (bf16), then for every stream j
//   R'[t][j][c] = bf16(sum_i comb[i][j] R[t][i][c] + post[j] out[c])
// with post and comb finished before the layer (the sublayer's pre ran in the
// serving framework), graph tensors: h_post [T][HC], h_res [T][HC HC] fp32
// (comb[i][j] at i HC + j, SGLang's MHCState). With counters (one per token),
// task (q, t) adds 1 to token t's once its columns of R' are written.
#pragma once
#include "../core.cuh"
#include "hc_pre.cuh"

namespace static_mk {

template <size_t SOFF, int H, int HC, int Q> // SOFF: the sent sums' offset in
                                             // the exchange region
__device__ __forceinline__ void hc_post_task(G const &g,
                                             float const *h_post,
                                             float const *h_res,
                                             __nv_bfloat16 const *res,
                                             __nv_bfloat16 *rout,
                                             uint32_t *done_cnt,
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
  constexpr int CB = HC * HC;
  // finished post and comb, graph tensors
  if (threadIdx.x < CB) {
    coef_s[2 * HC + threadIdx.x] = h_res[(size_t)t * CB + threadIdx.x];
  } else if (threadIdx.x < CB + HC) {
    coef_s[HC + threadIdx.x - CB] = h_post[(size_t)t * HC + threadIdx.x - CB];
  }
  if (threadIdx.x == 0) {
    STAGE_STAMP_AT(0)
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
        // rounded as SGLang's _mhc_post_torch: each product, the sum over i
        // in order, then post out added (no FMA)
        auto rik = [&](int i) {
          return __bfloat162float(
              reinterpret_cast<__nv_bfloat16 const *>(&ru[i])[k]);
        };
        float a = __fmul_rn(comb[0][j], rik(0));
#pragma unroll
        for (int i = 1; i < HC; i++) {
          a = __fadd_rn(a, __fmul_rn(comb[i][j], rik(i)));
        }
        ob[k] = __float2bfloat16(__fadd_rn(__fmul_rn(post[j], s[k]), a));
      }
      *reinterpret_cast<uint4 *>(rout + ((size_t)t * HC + j) * H + c) = ou;
    }
    if constexpr (REARM) {
      allreduce_rearm<S_RANK>(g, row, v);
    }
  }
  __syncthreads();
  if (done_cnt != nullptr && threadIdx.x == 0) {
    cnt_add(done_cnt, 1); // release: this task's columns of R'[t]
  }
}

// hc_post task (part x, token y). SEND: the allreduce_send node whose output it
// adds (params {H, 1}); RES, HRES: the streams' and h_res's producers (graph
// inputs); PARAMS {H, HC}; BUF_SLOTS {the residual streams R, the output R'
// ([T][HC H] bf16 graph tensors), h_post, h_res}; counters: none, or one per
// token
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class SEND,
          class RES,
          class HRES,
          class... IN>
__device__ __forceinline__ void run_hc_post(Maps const &,
                                            G const &g,
                                            KernelLocals &L,
                                            StaticTask const &tk) {
  constexpr int H = PARAMS::v[0], HC = PARAMS::v[1];
  static_assert(SEND::buf >= 0 && SEND::v[0] == H && SEND::v[1] >= 1,
                "hc_post: rows of H sent by an allreduce_send node (a "
                "token's rows of its input added: one row per token sent)");
  static_assert(
      exchange_bytes[SEND::buf] == (size_t)GPUS * T * H * 2,
      "hc_post: the sent sums, [GPUS][T][H] bf16 in the exchange region");
  static_assert(SELF::y == T, "hc_post: grid (parts, T tokens)");
  hc_post_task<exchange_offset[SEND::buf], H, HC, SELF::x>(
      g,
      buf_at<float const>(g, slot_at<BUF_SLOTS, 2>()),
      buf_at<float const>(g, slot_at<BUF_SLOTS, 3>()),
      buf_at<__nv_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<__nv_bfloat16>(g, slot_at<BUF_SLOTS, 1>()),
      SELF::counter >= 0 ? cnt_now(g) + SELF::counter + tk.y : nullptr,
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
