// tasks/moe_attn_residual.cuh (HIP, gfx950) -- the end of a K3 MoE layer with
// the next layer's aggregation 1 (SGLang ROCm: attn_res_hip's _agg_kernel after
// a MoE layer, prefix = y, its RMSNorm with the next layer's input_layernorm):
// residual_add's y = bf16(bf16(o + S) + prefix) and attn_residual's
// aggregation of the bank rows and y (attn_residual_task MOE). Task (x, y) of
// grid (T, NC, 1) = token x, columns [y CW, (y + 1) CW), CW = H / NC = 1024.
#pragma once
#include "attn_residual.cuh"
#include "sum_send.cuh" // the layout of its input o

namespace static_mk {

// SSUM: the addend's node (sum_gpus), O: the input's node (o in the exchange
// region, sum_send's layout); params = [H, eps bits]; buffer slots = bank,
// prefix, q, gain, ow, y, info, the partial sums' scratch, out (the next
// layer's input, [T][H] bf16); counters: 1 + 2 T (counter 0: the tasks done)
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class SSUM,
          class O,
          class... IN>
__device__ __forceinline__ void run_moe_attn_residual(Maps const &,
                                                      G const &g,
                                                      KernelLocals &L,
                                                      StaticTask const &tk) {
  constexpr int H = PARAMS::v[0];
  static_assert(SSUM::buf >= 0 && O::buf >= 0 && exchange_bytes[O::buf] == (size_t)GPUS * o_rank_bytes(H),
                "moe_attn_residual: S from sum_gpus's buffer, o from sum_send's layout");
  static_assert(SELF::x == T && SELF::counter >= 0 && H / SELF::y == AR_CW,
                "moe_attn_residual: one task per token and 1024-column chunk, its counters");
  attn_residual_task<exchange_offset[O::buf], H, SELF::y, 0, true>(
      g, L,
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 0>()),
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 2>()),
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 3>()),
      buf_at<__hip_bfloat16 const>(g, slot_at<BUF_SLOTS, 4>()),
      buf_at<__hip_bfloat16>(g, slot_at<BUF_SLOTS, 5>()),
      buf_at<int const>(g, slot_at<BUF_SLOTS, 6>()),
      __int_as_float(PARAMS::v[1]),
      buf_at<__hip_bfloat16>(g, slot_at<BUF_SLOTS, 8>()),
      buf_at<float>(g, slot_at<BUF_SLOTS, 7>()),
      cnt_of(g) + SELF::counter,
      tk.x,
      tk.y STAGE_STAMP_ARG(L.stage_stamps),
      buf_at<__hip_bfloat16 const>(g, SSUM::buf));
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_moe_attn_residual() {
  return (4 * AR_PW + AR_PW + 32) * 4 + GPUS * AR_CW * 2; // (attn_residual's LDS layout: the landing area unused)
}

} // namespace static_mk
