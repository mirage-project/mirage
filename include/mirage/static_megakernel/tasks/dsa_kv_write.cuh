// tasks/dsa_kv_write.cuh -- Write the MLA latent cache row of this step
// (GLM-5.3 / 5.2). Task t is token t at position p = seq_lens[t] - 1:
//   cache[row(p)] = [bf16(RMSNorm(kv_c) * gamma) (512) | bf16(RoPE(k_pe)) (64)]
// kv_c is columns [KV0, KV0 + 512) and k_pe is columns [KV0 + 512, KV0 + 576)
// of the fused A projection (a COMBINE_ADD GEMM node; the demo uses a
// scaled_gemm_tile). The task rounds kv_c and k_pe to bf16, as a linear layer
// does. The RMSNorm and RoPE math is fp32. The task rounds each result to bf16
// one time, in the cache write. RoPE is interleaved (dsa_common.cuh).
//   row(p) = block_table[t][p / PAGE] * PAGE + p % PAGE
// A token with seq_lens[t] <= 0 (an idle slot) writes no row. A token with
// seq_lens[t] > MAXP x PAGE writes no row (its page is not in the block
// table).
#pragma once
#include "../dsa_common.cuh"

namespace static_mk {

// Task x is token x. PARAMS {KV0, PAGE, MAXP, eps (fp32 bits)}. BUF_SLOTS
// {gamma (fp32 [512]), cos_sin, seq_lens, block_table, cache ([rows][576]
// bf16)}. QKVA: the fused A projection.
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class QKVA,
          class... IN>
__device__ __forceinline__ void run_dsa_kv_write(Maps const &,
                                                 G const &g,
                                                 KernelLocals &L,
                                                 StaticTask const &tk) {
  constexpr int KV0 = PARAMS::v[0], PAGE = PARAMS::v[1], MAXP = PARAMS::v[2];
  constexpr int N = QKVA::v[2], KVL = 512, ROPE = 64, D = KVL + ROPE;
  constexpr int GAMMA = slot_at<BUF_SLOTS, 0>(), CS = slot_at<BUF_SLOTS, 1>(),
                SEQ = slot_at<BUF_SLOTS, 2>(), BT = slot_at<BUF_SLOTS, 3>(),
                CACHE = slot_at<BUF_SLOTS, 4>();
  static_assert(KV0 + D <= N && KV0 % 2 == 0 && N % 2 == 0,
                "dsa_kv_write: kv_c and k_pe in the A projection (aligned "
                "vector loads)");
  static_assert(SELF::x == T && SELF::counter >= 0,
                "dsa_kv_write: a task per token, a count of its tasks");
  int const t = tk.x;
  __syncthreads(); // wait until the previous task completes its shared-memory
                   // reads
  if (threadIdx.x < 32) {
    dsa::fx_wait_cols<QKVA>(g, KV0, KV0 + D);
  }
  __syncthreads();
  dsa::stamp_ready(L);
  int const pos = buf_at<int const>(g, SEQ)[t] - 1;
  if (pos >= 0 && pos < MAXP * PAGE) { // the same value in all threads
    auto const *qa = buf_at<unsigned long long const>(g, QKVA::buf);
    int const page = buf_at<int const>(g, BT)[(size_t)t * MAXP + pos / PAGE];
    __nv_bfloat16 *row = buf_at<__nv_bfloat16>(g, CACHE) +
                         ((size_t)page * PAGE + pos % PAGE) * D;
    dsa::fx_rmsnorm<KVL>(qa,
                         N,
                         t,
                         KV0,
                         buf_at<float const>(g, GAMMA),
                         __int_as_float(PARAMS::v[3]),
                         row);
    // k_pe: thread i (i < 32) takes pair i
    if (threadIdx.x < ROPE / 2) {
      int const i = threadIdx.x;
      dsa::fx_rope_store(qa,
                         N,
                         t,
                         KV0 + KVL + 2 * i,
                         buf_at<__nv_bfloat16 const>(g, CS),
                         pos,
                         i,
                         row + KVL + 2 * i);
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
constexpr int smem_dsa_kv_write() {
  return 0;
}

} // namespace static_mk
