// tasks/sm90_rmsnorm.cuh -- the sm90_rmsnorm task type: MPK's Hopper RMSNorm
// (rmsnorm_hopper.cuh) of one token. Task x of grid (T, 1, 1): row x.
#pragma once
#include "sm90_common.cuh"
#include "tasks/hopper/rmsnorm_hopper.cuh"

namespace static_mk {

// PARAMS {H, eps (fp32 bits)}; BUF_SLOTS {input [T, H] bf16, weight [H] bf16,
// output [T, H] bf16}
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_sm90_rmsnorm(Maps const &,
                                                 G const &g,
                                                 KernelLocals &,
                                                 StaticTask const &tk) {
  constexpr int H = PARAMS::v[0];
  sm90_task_begin<IN...>(g);
  using bf16 = cute::bfloat16_t;
  kernel::rms_norm_hopper_impl<bf16, 1, H>(
      buf_at<bf16 const>(g, slot_at<BUF_SLOTS, 0>()) + (size_t)tk.x * H,
      buf_at<bf16 const>(g, slot_at<BUF_SLOTS, 1>()),
      buf_at<bf16>(g, slot_at<BUF_SLOTS, 2>()) + (size_t)tk.x * H,
      __int_as_float(PARAMS::v[1]));
  sm90_task_end<SELF>(g);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_sm90_rmsnorm() {
  return SM90_BODY_SMEM;
}

} // namespace static_mk
