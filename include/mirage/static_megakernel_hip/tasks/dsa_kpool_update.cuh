// tasks/dsa_kpool_update.cuh (HIP, gfx950) -- the pooled indexer keys of GLM-5.3-Flash's DSA lightning indexer
// (index_kpool = 4), the math of include/mirage/static_megakernel/tasks/dsa_kpool_update.cuh on SGLang ROCm's buffers
// (kpool_fp8_index.py _kpool_decode_update_and_maybe_write_cache_kernel; amd/notes/sglang_v0522_glm_dsa_contract.md
// 1.4 step 6, 2.2, 2.3). Task t = token t at position p = seq_lens[t] - 1 of the request in slot r = req_slots[t]:
//   k     columns [0, 128) of the indexer projection x [wk | gate | weights_proj]^T (a gemm_tile node's K parts in slots),
//         bf16
//   kn    bf16(LayerNorm(k) w + b) (fp32, two-pass mean / variance)
//   g     columns [128, 256), bf16
//   p % 4 == 3: pool j = p / 4 closes: per dim the softmax over the 4 slots of (gate + ape[slot]) weights the keys (slot
//         order; the first 3 from the tail ring, slot p from registers) -> bf16 -> Hadamard-128 -> bf16 -> e4m3 with a
//         UE8M0 scale -> pooled page pb = pooled_bt[t][j / 64] (SGLang: real_page_table[b, (j / 64) 4] / 4), key j % 64
//         in the preshuffled layout (dsa_util.cuh)
//   then (kn, g) -> tail slot p % RING of request r ([R][RING][128] bf16 each)
// 32 lanes of wave 0 do the work (lane l: dims 4 l .. + 3); stores at device scope (dsa_index_score reads the page after
// the node's count). An idle row (seq_lens <= 0, req_slots < 0) writes nothing. One count per task.
#pragma once
#include "../dsa_util.cuh"

namespace static_mk {

// PARAMS {PAGE (64), MAXPP (pooled block table columns), RING, eps (fp32 bits)}; BUF_SLOTS {w, b (fp32 [128]), ape (fp32
// [4][128]), seq_lens (int32 [T]), req_slots (int64 [T]), pooled_bt (int32 [T][MAXPP]), tail_k, tail_score (bf16
// [R][RING][128]), the pooled cache (uint8 [pages][8448])}; KW: the indexer projection (gemm_tile, K parts in slots)
template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class KW, class... IN>
__device__ __forceinline__ void run_dsa_kpool_update(Maps const &, G const &g, KernelLocals &L, StaticTask const &tk) {
  constexpr int PAGE = PARAMS::v[0], MAXPP = PARAMS::v[1], RING = PARAMS::v[2], NA = KW::v[2], KPOOL = 4;
  static_assert(KW::buf >= 0 && KW::v[0] == COMBINE_SLOTS && NA >= 256 && NA % 4 == 0,
                "dsa_kpool_update: k and the gate in columns [0, 256) of a gemm_tile node's K parts in slots");
  static_assert(PAGE == dsa::IDX_PAGE_KEYS && RING >= KPOOL, "dsa_kpool_update: SGLang's page of 64 pools, a ring of 4+");
  static_assert(SELF::x == T && SELF::counter >= 0, "dsa_kpool_update: a task per token, a count of its tasks");
  int const t = tk.x, tid = STATIC_TIDX;
  int const pos = buf_at<int const>(g, slot_at<BUF_SLOTS, 3>())[t] - 1;
  int const r = (int)buf_at<long long const>(g, slot_at<BUF_SLOTS, 4>())[t];
  if (tid < 32 && pos >= 0 && r >= 0 && pos / KPOOL < MAXPP * PAGE) {
    int const lane = tid, d0 = 4 * lane;
    float const *parts = buf_at<float const>(g, KW::buf);
    float4 const k4 = dsa::slots_col4<KW::y, NA>(parts, t, d0), g4 = dsa::slots_col4<KW::y, NA>(parts, t, 128 + d0);
    float k[4] = {round_bf16(k4.x), round_bf16(k4.y), round_bf16(k4.z), round_bf16(k4.w)};
    float const gt[4] = {round_bf16(g4.x), round_bf16(g4.y), round_bf16(g4.z), round_bf16(g4.w)};
    float sum = (k[0] + k[1]) + (k[2] + k[3]);
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
      sum += __shfl_xor(sum, d, 32);
    }
    float const mean = sum * (1.0f / 128);
    float var = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      var += (k[j] - mean) * (k[j] - mean);
    }
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
      var += __shfl_xor(var, d, 32);
    }
    float const rstd = rsqrtf(var * (1.0f / 128) + __int_as_float(PARAMS::v[3]));
    float const *w = buf_at<float const>(g, slot_at<BUF_SLOTS, 0>()), *b = buf_at<float const>(g, slot_at<BUF_SLOTS, 1>());
#pragma unroll
    for (int j = 0; j < 4; j++) {
      k[j] = round_bf16((k[j] - mean) * rstd * w[d0 + j] + b[d0 + j]);
    }
    size_t const row = (size_t)r * RING * 128;
    __hip_bfloat16 *const tk_r = buf_at<__hip_bfloat16>(g, slot_at<BUF_SLOTS, 6>()) + row;
    __hip_bfloat16 *const ts_r = buf_at<__hip_bfloat16>(g, slot_at<BUF_SLOTS, 7>()) + row;
    if (pos % KPOOL == KPOOL - 1) {
      float const *ape = buf_at<float const>(g, slot_at<BUF_SLOTS, 2>());
      float sc[KPOOL][4], kv[KPOOL][4], mx[4];
#pragma unroll
      for (int s = 0; s < KPOOL; s++) {
        int const ps = (pos - (KPOOL - 1) + s) % RING;
        bool const cur = s == KPOOL - 1;
        uint2 const tkw = cur ? make_uint2(0u, 0u) : *reinterpret_cast<uint2 const *>(tk_r + ps * 128 + d0);
        uint2 const tsw = cur ? make_uint2(0u, 0u) : *reinterpret_cast<uint2 const *>(ts_r + ps * 128 + d0);
        float const tkv[4] = {bf16_lo(tkw.x), bf16_hi(tkw.x), bf16_lo(tkw.y), bf16_hi(tkw.y)};
        float const tsv[4] = {bf16_lo(tsw.x), bf16_hi(tsw.x), bf16_lo(tsw.y), bf16_hi(tsw.y)};
#pragma unroll
        for (int j = 0; j < 4; j++) {
          kv[s][j] = cur ? k[j] : tkv[j];
          sc[s][j] = (cur ? gt[j] : tsv[j]) + ape[s * 128 + d0 + j];
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
        x[0][j] = round_bf16(acc / den);
      }
      dsa::hadamard128<1>(x, lane);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        x[0][j] = round_bf16(x[0][j]);
      }
      uint32_t q8[1];
      int e[1];
      dsa::quant128<1>(x, q8, e);
      int const pool = pos / KPOOL;
      int const pb = buf_at<int const>(g, slot_at<BUF_SLOTS, 5>())[(size_t)t * MAXPP + pool / PAGE];
      unsigned char *const pg = buf_at<unsigned char>(g, slot_at<BUF_SLOTS, 8>()) + (size_t)pb * dsa::IDX_PAGE_BYTES;
      int const o = pool % PAGE;
      dsa::st4_dev(pg + dsa::idx_key_off(o, d0), q8[0]);
      if (lane == 0) {
        dsa::st4_dev(pg + dsa::IDX_SCALE_OFF + 4 * o, __float_as_uint(dsa::exp2_int(e[0])));
      }
    }
    // this token into its tail slot (after the reads of the other slots)
    int const ps = pos % RING;
    asm volatile("global_store_dwordx2 %0, %1, off sc1" ::"v"(tk_r + ps * 128 + d0),
                 "v"(v2u{pack_bf16x2(k[0], k[1]), pack_bf16x2(k[2], k[3])})
                 : "memory");
    asm volatile("global_store_dwordx2 %0, %1, off sc1" ::"v"(ts_r + ps * 128 + d0),
                 "v"(v2u{pack_bf16x2(gt[0], gt[1]), pack_bf16x2(gt[2], gt[3])})
                 : "memory");
  }
  task_post(cnt_of(g) + SELF::counter, 1);
}

template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class... IN>
constexpr int smem_dsa_kpool_update() {
  return 0;
}

} // namespace static_mk
