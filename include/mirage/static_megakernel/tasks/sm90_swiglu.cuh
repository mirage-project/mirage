// tasks/sm90_swiglu.cuh -- the sm90_swiglu task type: MPK's clamped SwiGLU
// (clamped_swiglu.cuh) of one token's top-k slots. Task x of grid (T, 1, 1).
#pragma once
#include "sm90_common.cuh"
#include "tasks/ampere/clamped_swiglu.cuh"

namespace static_mk {

// PARAMS {I, TOPK, limit (fp32 bits), alpha (fp32 bits)}; BUF_SLOTS {input
// [T, TOPK, 2 I] bf16 (gate then up), output [T, TOPK, I] bf16}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_sm90_swiglu(Maps const &,
                                                G const &g,
                                                KernelLocals &,
                                                StaticTask const &tk) {
  constexpr int I = PARAMS::v[0], TOPK = PARAMS::v[1];
  sm90_task_begin<IN...>(g);
  kernel::clamped_swiglu_task_impl<cute::bfloat16_t, TOPK, I, 2 * I, I>(
      buf_at<cute::bfloat16_t const>(g, slot_at<BUF_SLOTS, 0>()) +
          (size_t)tk.x * TOPK * 2 * I,
      buf_at<cute::bfloat16_t>(g, slot_at<BUF_SLOTS, 1>()) +
          (size_t)tk.x * TOPK * I,
      __int_as_float(PARAMS::v[2]),
      __int_as_float(PARAMS::v[3]),
      TOPK);
  sm90_task_end<SELF>(g);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_sm90_swiglu() {
  return 0;
}

} // namespace static_mk
