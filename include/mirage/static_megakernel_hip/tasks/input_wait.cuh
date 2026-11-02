// tasks/input_wait.cuh (HIP, gfx950) -- a layer input written by the kernel
// before it while this one already runs (two layer kernels overlapped on two
// streams, the AMD stand-in for programmatic dependent launch): task x of grid
// (T, 1, 1) = token x polls its row of the input ([T][H] bf16 graph tensor,
// all-ones until written: no 4-B word of real bf16 data is all-ones) until
// every word is written, copies it into the node's buffer (device scope; this
// kernel's readers wait for the node's count), and, with params[1] (re-arm),
// writes all-ones back into the input for the next launch (no other reader of
// it in this kernel). One count per task.
#pragma once
#include "../core.cuh"

namespace static_mk {

template <int H>
__device__ __forceinline__ void input_wait_task(__hip_bfloat16 *a,
                                                __hip_bfloat16 *a_out,
                                                bool rearm,
                                                uint32_t *cnt,
                                                int t) {
  constexpr int NV = H / 8; // 16-B vectors per row
  constexpr int PER = (NV + 255) / 256;
  uint4 va[PER];
  bool ok[PER];
#pragma unroll
  for (int q = 0; q < PER; q++) {
    ok[q] = (int)STATIC_TIDX + 256 * q >= NV;
  }
  for (;;) {
    bool all = true;
#pragma unroll
    for (int q = 0; q < PER; q++) {
      if (!ok[q]) {
        va[q] = ld16_at<AUX_DEV>(a, (uint32_t)(((size_t)t * H + 8 * ((int)STATIC_TIDX + 256 * q)) * 2));
        ok[q] = valid16w(va[q]);
      }
      all = all && ok[q];
    }
    if (__all(all)) {
      break;
    }
    tail_pause();
  }
#pragma unroll
  for (int q = 0; q < PER; q++) {
    int const v = (int)STATIC_TIDX + 256 * q;
    if (v < NV) {
      size_t const e = (size_t)t * H + 8 * v;
      st16_dev(a_out + e, va[q]);
      if (rearm) {
        st16_dev(a + e, make_uint4(~0u, ~0u, ~0u, ~0u));
      }
    }
  }
  task_post(cnt, 1);
}

// task x = token x; params = [H, re-arm]; buffer slots = the input (a graph tensor); the node's buffer = the copy
template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
__device__ __forceinline__ void run_input_wait(Maps const &,
                                               G const &g,
                                               KernelLocals &,
                                               StaticTask const &tk) {
  constexpr int H = PARAMS::v[0];
  static_assert(SELF::x == T && SELF::buf >= 0 && SELF::counter >= 0 && H % 8 == 0,
                "input_wait: one task per token, its output buffer and counter");
  input_wait_task<H>(buf_at<__hip_bfloat16>(g, slot_at<BUF_SLOTS, 0>()),
                     buf_at<__hip_bfloat16>(g, SELF::buf),
                     PARAMS::v[1] != 0,
                     cnt_of(g) + SELF::counter,
                     tk.x);
}

template <class SELF,
          class PARAMS,
          class BUF_SLOTS,
          class MAP_SLOTS,
          class... IN>
constexpr int smem_input_wait() {
  return 0;
}

} // namespace static_mk
