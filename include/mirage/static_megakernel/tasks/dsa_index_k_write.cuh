// tasks/dsa_index_k_write.cuh -- Write the DSA indexer key of this step
// (GLM-5.3 / 5.2 "full" layers). Task t is token t at position
// p = seq_lens[t] - 1:
//   k   Columns [0, 128) of the indexer projection x [wk | weights_proj]^T,
//       rounded to bf16. A COMBINE_ADD GEMM node makes this projection (the
//       demo uses a bf16 gemm_tile).
//   kn  LayerNorm(k) * w + b in fp32 (two-pass mean and variance, eps from
//       PARAMS). Then RoPE (interleaved, dsa_common.cuh) on dims [0, 64) of
//       kn, in fp32.
//   q8  e4m3(kn / s). s is the UE8M0 scale of the 128 values.
// A page of the index cache holds PAGE keys. First come the 128 e4m3 bytes of
// each key. Then come the PAGE scales. Each scale is 4 copies of the UE8M0
// byte. Position o of the page is at:
//   values  page base + o * 128
//   scale   page base + PAGE * 128 + o * 4
// One warp does the work. Lane l takes dims 4 l .. 4 l + 3. A token with
// seq_lens[t] <= 0 (an idle slot) writes no key. A token with seq_lens[t] >
// MAXP x PAGE writes no key (its page is not in the block table).
#pragma once
#include "../dsa_common.cuh"

namespace static_mk {

// Task x is token x. PARAMS {PAGE, MAXP, eps (fp32 bits)}. BUF_SLOTS {w (fp32
// [128]), b (fp32 [128]), cos_sin, seq_lens, block_table, index cache (uint8,
// [pages][PAGE x 132])}. KW: the indexer projection (a COMBINE_ADD GEMM).
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class KW,
          class... IN>
__device__ __forceinline__ void run_dsa_index_k_write(Maps const &,
                                                      G const &g,
                                                      KernelLocals &L,
                                                      StaticTask const &tk) {
  constexpr int PAGE = PARAMS::v[0], MAXP = PARAMS::v[1], N = KW::v[2];
  constexpr int W = slot_at<BUF_SLOTS, 0>(), B = slot_at<BUF_SLOTS, 1>(),
                CS = slot_at<BUF_SLOTS, 2>(), SEQ = slot_at<BUF_SLOTS, 3>(),
                BT = slot_at<BUF_SLOTS, 4>(), ICACHE = slot_at<BUF_SLOTS, 5>();
  static_assert(N >= 128, "dsa_index_k_write: k in the first 128 columns");
  static_assert(SELF::x == T && SELF::counter >= 0,
                "dsa_index_k_write: a task per token, a count of its tasks");
  int const t = tk.x;
  // The task uses no shared memory, so it starts with no barrier.
  if (threadIdx.x < 32) {
    dsa::fx_wait_cols<KW>(g, 0, 128);
  }
  __syncthreads();
  dsa::stamp_ready(L);
  int const pos = buf_at<int const>(g, SEQ)[t] - 1;
  if (threadIdx.x < 32 && pos >= 0 && pos < MAXP * PAGE) {
    int const lane = threadIdx.x, d0 = 4 * lane;
    auto const *kw = buf_at<unsigned long long const>(g, KW::buf);
    float k[1][4], sum = 0.f; // one vector for rope_quant128
#pragma unroll
    for (int j = 0; j < 4; j++) {
      k[0][j] = dsa::fx_bf16(kw, N, t, d0 + j);
      sum += k[0][j];
    }
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
      sum += __shfl_xor_sync(0xffffffffu, sum, d);
    }
    float const mean = sum * (1.0f / 128);
    float var = 0.f;
#pragma unroll
    for (int j = 0; j < 4; j++) {
      var += (k[0][j] - mean) * (k[0][j] - mean);
    }
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
      var += __shfl_xor_sync(0xffffffffu, var, d);
    }
    float const rstd =
        rsqrtf(var * (1.0f / 128) + __int_as_float(PARAMS::v[2]));
    float const *w = buf_at<float const>(g, W), *b = buf_at<float const>(g, B);
#pragma unroll
    for (int j = 0; j < 4; j++) {
      k[0][j] = (k[0][j] - mean) * rstd * w[d0 + j] + b[d0 + j];
    }
    float c0 = 1.f, s0 = 0.f, c1 = 1.f, s1 = 0.f;
    dsa::rope_cs2(
        buf_at<__nv_bfloat16 const>(g, CS), pos, lane, c0, s0, c1, s1);
    uint32_t q[1];
    int e[1];
    dsa::rope_quant128<1>(k, lane, c0, s0, c1, s1, q, e);
    int const page = buf_at<int const>(g, BT)[(size_t)t * MAXP + pos / PAGE];
    unsigned char *pg = buf_at<unsigned char>(g, ICACHE) +
                        (size_t)page * dsa::index_page_bytes(PAGE);
    int const o = pos % PAGE;
    *reinterpret_cast<uint32_t *>(pg + o * 128 + d0) = q[0];
    if (lane == 0) {
      *reinterpret_cast<uint32_t *>(pg + PAGE * 128 + o * 4) =
          (uint32_t)e[0] * 0x01010101u;
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    cnt_add(g.cnt + SELF::counter, 1);
  }
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_dsa_index_k_write() {
  return 0;
}

} // namespace static_mk
