// tasks/sm90_mul_sum_add.cuh -- the sm90_mul_sum_add task type: MPK's MoE
// combine (mul_sum_add_sm100.cuh, architecture-neutral): one token's top-k
// expert outputs weighted, summed, plus the residual. Task x of grid (T, 1, 1).
#pragma once
#include "sm90_common.cuh"
#include "tasks/blackwell/mul_sum_add_sm100.cuh"

namespace static_mk {

// PARAMS {H, TOPK}; BUF_SLOTS {expert outputs [T, TOPK, H] bf16, weights [T,
// TOPK] fp32, residual [T, H] bf16, output [T, H] bf16}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_sm90_mul_sum_add(Maps const &,
                                                     G const &g,
                                                     KernelLocals &,
                                                     StaticTask const &tk) {
  constexpr int H = PARAMS::v[0], TOPK = PARAMS::v[1];
  sm90_task_begin<IN...>(g);
  using bf16 = cute::bfloat16_t;
  kernel::mul_sum_add_sm100_task_impl<bf16, 1, H, TOPK, H>(
      buf_at<bf16 const>(g, slot_at<BUF_SLOTS, 0>()) + (size_t)tk.x * TOPK * H,
      buf_at<float const>(g, slot_at<BUF_SLOTS, 1>()) + (size_t)tk.x * TOPK,
      buf_at<bf16 const>(g, slot_at<BUF_SLOTS, 2>()) + (size_t)tk.x * H,
      buf_at<bf16>(g, slot_at<BUF_SLOTS, 3>()) + (size_t)tk.x * H);
  sm90_task_end<SELF>(g);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_sm90_mul_sum_add() {
  return 0;
}

} // namespace static_mk
