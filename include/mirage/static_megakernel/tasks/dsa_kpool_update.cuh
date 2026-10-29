// tasks/dsa_kpool_update.cuh -- Update the pooled indexer keys of the DSA
// lightning indexer (GLM-5.3-Flash, index_kpool = 4). Task t is token t at
// position p = seq_lens[t] - 1 of the request in slot r = req_slots[t] (int64:
// SGLang's req_pool_indices, bound as it is):
//   k     Columns [0, 128) of the indexer projection x [wk | gate |
//         weights_proj]^T, rounded to bf16. A COMBINE_ADD GEMM node makes this
//         projection (the demo uses a bf16 gemm_tile).
//   kn    bf16(LayerNorm(k) * w + b), fp32 math (two-pass mean and variance,
//         eps from PARAMS).
//   g     Columns [128, 256), the compress gate x gate^T, rounded to bf16.
//   tail  The request's last keys and gates: tail_k[r][s], tail_score[r][s]
//         (bf16 [R][RING][128]) hold the token at position q in slot
//         s = q % RING.
// When p % 4 = 3, the task closes pool j = p / 4 (positions 4 j .. 4 j + 3):
// for each dim, the softmax over the 4 positions of (gate + ape[slot]) weights
// the keys (added in slot order), then bf16 -> Hadamard-128 -> bf16 -> e4m3
// with the UE8M0 scale s (dsa_common hadamard128, quant128). The tail gives
// the first 3 positions, the registers give p. Page pb = pooled_bt[t][j / PAGE]
// of the pooled cache holds PAGE pooled keys: the 128 e4m3 bytes of pool
// slot o = j % PAGE at pb * PAGE * 132 + o * 128, its fp32 scale s at
// pb * PAGE * 132 + PAGE * 128 + o * 4. Then the task writes (kn, g) to slot
// p % RING of the tail. These are the steps and the layouts of SGLang's
// kpool_decode_update_and_maybe_write_cache (kpool_fp8_index.py), on SGLang's
// own buffers: the index-K buffer of the layer, the tail buffers of the layer,
// pooled_bt = its pooled page table (64 pools a page).
// One warp does the work. Lane l takes dims 4 l .. 4 l + 3. An idle slot
// (seq_lens[t] <= 0 or req_slots[t] < 0) writes nothing. Each slot t must be a
// different request (the tail), unless NT (PARAMS[4]) > 1: then task x is the
// NT consecutive tokens x NT .. x NT + NT - 1 of one request, in order (a
// speculative verify step: the later tokens read the tail entries the earlier
// ones wrote, and the pools they close); grid x = T / NT.
#pragma once
#include "../dsa_common.cuh"

namespace static_mk {

// Task x is token x (NT = 1) or tokens x NT .. of one request. PARAMS {PAGE
// (pools in a page), MAXPP (pooled block table columns), RING (tail slots), eps
// (fp32 bits)[, NT (tokens per task, default 1)]}. BUF_SLOTS {w (fp32 [128]),
// b (fp32 [128]), ape (fp32 [4][128]), seq_lens, req_slots (int64 [T]),
// pooled_bt (int32 [T][MAXPP]), tail_k, tail_score, the pooled cache (uint8
// [pages][PAGE x 132])}. KW: the indexer projection (a COMBINE_ADD GEMM).
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class KW,
          class... IN>
__device__ __forceinline__ void run_dsa_kpool_update(Maps const &,
                                                     G const &g,
                                                     KernelLocals &L,
                                                     StaticTask const &tk) {
  constexpr int PAGE = PARAMS::v[0], MAXPP = PARAMS::v[1], RING = PARAMS::v[2],
                N = KW::v[2], KPOOL = 4;
  constexpr int NT = PARAMS::v[4] > 0 ? PARAMS::v[4] : 1; // tokens per task
  constexpr int W = slot_at<BUF_SLOTS, 0>(), B = slot_at<BUF_SLOTS, 1>(),
                APE = slot_at<BUF_SLOTS, 2>(), SEQ = slot_at<BUF_SLOTS, 3>(),
                REQ = slot_at<BUF_SLOTS, 4>(), PBT = slot_at<BUF_SLOTS, 5>(),
                TK = slot_at<BUF_SLOTS, 6>(), TS = slot_at<BUF_SLOTS, 7>(),
                PCACHE = slot_at<BUF_SLOTS, 8>();
  static_assert(N >= 256,
                "dsa_kpool_update: k and the gate in columns [0, 256)");
  static_assert(RING >= KPOOL && PAGE % 4 == 0,
                "dsa_kpool_update: the tail holds a pool");
  static_assert(SELF::x * NT == T && SELF::counter >= 0,
                "dsa_kpool_update: a task per token (or per NT tokens of a "
                "request), a count of its tasks");
  // The task uses no shared memory, so it starts with no barrier.
  if (threadIdx.x < 32) {
    dsa::fx_wait_cols<KW>(g, 0, 256);
  }
  __syncthreads();
  dsa::stamp_ready(L);
#pragma unroll 1
  for (int i = 0; i < NT; i++) { // the request's tokens in order (NT > 1)
    int const t = tk.x * NT + i;
    int const pos = buf_at<int const>(g, SEQ)[t] - 1,
              r = (int)buf_at<long long const>(g, REQ)[t];
    if (threadIdx.x < 32 && pos >= 0 && r >= 0 && pos / KPOOL < MAXPP * PAGE) {
      int const lane = threadIdx.x, d0 = 4 * lane;
      auto const *kw = buf_at<unsigned long long const>(g, KW::buf);
      float k[4], gt[4], sum = 0.f;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        k[j] = dsa::fx_bf16(kw, N, t, d0 + j);
        gt[j] = dsa::fx_bf16(kw, N, t, 128 + d0 + j);
        sum += k[j];
      }
#pragma unroll
      for (int d = 16; d > 0; d >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, d);
      }
      float const mean = sum * (1.0f / 128);
      float var = 0.f;
#pragma unroll
      for (int j = 0; j < 4; j++) {
        var += (k[j] - mean) * (k[j] - mean);
      }
#pragma unroll
      for (int d = 16; d > 0; d >>= 1) {
        var += __shfl_xor_sync(0xffffffffu, var, d);
      }
      float const rstd =
          rsqrtf(var * (1.0f / 128) + __int_as_float(PARAMS::v[3]));
      float const *w = buf_at<float const>(g, W),
                  *b = buf_at<float const>(g, B);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        k[j] = dsa::bf16r((k[j] - mean) * rstd * w[d0 + j] + b[d0 + j]);
      }
      size_t const row = (size_t)r * RING * 128;
      __nv_bfloat16 *const tk_r = buf_at<__nv_bfloat16>(g, TK) + row;
      __nv_bfloat16 *const ts_r = buf_at<__nv_bfloat16>(g, TS) + row;
      if (pos % KPOOL == KPOOL - 1) {
        // close pool pos / 4: the scores of the 4 slots, their max, then the
        // weighted sum in slot order
        float const *ape = buf_at<float const>(g, APE);
        float sc[KPOOL][4], kv[KPOOL][4], mx[4];
#pragma unroll
        for (int s = 0; s < KPOOL; s++) {
          int const q = pos - (KPOOL - 1) + s, ps = q % RING;
#pragma unroll
          for (int j = 0; j < 4; j++) {
            bool const cur = s == KPOOL - 1;
            float const gs =
                cur ? gt[j] : __bfloat162float(ts_r[ps * 128 + d0 + j]);
            kv[s][j] = cur ? k[j] : __bfloat162float(tk_r[ps * 128 + d0 + j]);
            sc[s][j] = gs + ape[s * 128 + d0 + j];
            mx[j] = s == 0 ? sc[s][j] : fmaxf(mx[j], sc[s][j]);
          }
        }
        float x[1][4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
          float acc = 0.f, den = 0.f;
#pragma unroll
          for (int s = 0; s < KPOOL; s++) {
            float const p = __expf(sc[s][j] - mx[j]);
            den += p;
            acc += kv[s][j] * p;
          }
          x[0][j] = dsa::bf16r(acc / den);
        }
        dsa::hadamard128<1>(x, lane);
#pragma unroll
        for (int j = 0; j < 4; j++) {
          x[0][j] = dsa::bf16r(x[0][j]);
        }
        uint32_t q8[1];
        int e[1];
        dsa::quant128<1>(x, q8, e);
        int const pool = pos / KPOOL;
        int const pb =
            buf_at<int const>(g, PBT)[(size_t)t * MAXPP + pool / PAGE];
        unsigned char *pg = buf_at<unsigned char>(g, PCACHE) +
                            (size_t)pb * dsa::index_page_bytes(PAGE);
        int const o = pool % PAGE;
        *reinterpret_cast<uint32_t *>(pg + o * 128 + d0) = q8[0];
        if (lane == 0) {
          *reinterpret_cast<float *>(pg + PAGE * 128 + o * 4) =
              dsa::exp2_int(e[0]);
        }
      }
      // this token into its tail slot (after the reads of the other slots)
      int const ps = pos % RING;
      *reinterpret_cast<__nv_bfloat162 *>(tk_r + ps * 128 + d0) =
          __floats2bfloat162_rn(k[0], k[1]);
      *reinterpret_cast<__nv_bfloat162 *>(tk_r + ps * 128 + d0 + 2) =
          __floats2bfloat162_rn(k[2], k[3]);
      *reinterpret_cast<__nv_bfloat162 *>(ts_r + ps * 128 + d0) =
          __floats2bfloat162_rn(gt[0], gt[1]);
      *reinterpret_cast<__nv_bfloat162 *>(ts_r + ps * 128 + d0 + 2) =
          __floats2bfloat162_rn(gt[2], gt[3]);
    }
    if constexpr (NT > 1) { // the next token reads this one's tail entries
      __syncwarp();
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(cnt_now(g) + SELF::counter, 1);
  }
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_dsa_kpool_update() {
  return 0;
}

} // namespace static_mk
