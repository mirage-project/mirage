// tasks/dsa_o_absorb.cuh (HIP, gfx950) -- GLM-5.3-Flash DSA's output absorption (SGLang: forward_mla_rocm.py, the W_UV
// torch.bmm of a bf16 kv_b_proj): v[t][h][d] = bf16(sum_l attn[t][h][l] w_vc[h][d][l]), the same MFMA task as
// mla_absorb.cuh (K3's query absorption) with the input dims 512 (the latent) and the output dims 256 (v_head_dim). Task
// (x, y) of grid (heads, 256 / 128, 1) = head x, output rows [128 y, + 128):
//   attn = dsa_sparse_attn's output [T][heads x 512] bf16 (0xFF-prefilled, polled)
//   w_vc given as [heads][256][512] bf16 (kv_b_proj's v part per head, row-major as the checkpoint's)
//   -> v [T][heads x 256] bf16 (the node's buffer, 0xFF-prefilled: o_proj polls it)
#pragma once
#include "mla_absorb.cuh"

namespace static_mk {

// PARAMS {HEADS, 512 (in), 0, DV (out)}; BUF_SLOTS {w_vc}; A: dsa_sparse_attn (its buffer: attn)
template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class A, class... IN>
__device__ __forceinline__ void run_dsa_o_absorb(Maps const &, G const &g, KernelLocals &L, StaticTask const &tk) {
  static_assert(A::buf >= 0, "dsa_o_absorb: dsa_sparse_attn's output");
  static_assert(SELF::buf >= 0 && SELF::x == PARAMS::v[0] && SELF::y * 128 == PARAMS::v[3],
                "dsa_o_absorb: one task per head and 128 output rows, its output");
  mla_absorb_task<PARAMS::v[0], PARAMS::v[1], PARAMS::v[2], PARAMS::v[3]>(
      L,
      buf_at<__hip_bfloat16 const>(g, A::buf),
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<__hip_bfloat16>(g, SELF::buf),
      tk.x,
      tk.y);
  if constexpr (SELF::counter >= 0) { // (its consumer counted, not polled: o_proj loads the whole activation in one DMA pass)
    task_post(cnt_of(g) + SELF::counter, 1);
  }
}

template <class SELF, class PARAMS, class BUF_SLOTS, class MAP_SLOTS, class... IN>
constexpr int smem_dsa_o_absorb() {
  return T * PARAMS::v[1] * 2;
}

} // namespace static_mk
