// M3-I5c generality TU.
//
// The shipped call sites pass start_expert=0 / end_expert=NUM_EXPERTS as
// literals, so the compaction's `num_local_experts` folds to a constant. This
// TU passes them as RUNTIME values and pins __launch_bounds__ below
// NUM_EXPERTS, i.e. the multi-tile, expert-parallel-slice shape the fix must
// also handle. It exists to prove the general path compiles and does not
// spill; it is never launched.

#include "blackwell/topk_sigmoid_sm100.cuh"
#include "blackwell/topk_softmax_sm100.cuh"

using bf16 = cute::bfloat16_t;

// blockDim.x < NUM_EXPERTS -> the compaction takes several tiles.
template <int VPT, int EXPERTS, int NUM_ROWS, int BLOCK>
__global__ __launch_bounds__(BLOCK) void k_softmax_ep(void *in,
                                                      void *out,
                                                      void *routing,
                                                      void *active,
                                                      int k,
                                                      int start_expert,
                                                      int end_expert) {
  kernel::topk_softmax_task_impl<bf16, VPT, EXPERTS, /*WARPS_PER_CTA=*/8,
                                 /*BYTES_PER_LDG=*/16>(
      in, /*finished=*/nullptr, out, NUM_ROWS, k, routing, active,
      start_expert, end_expert, /*renormalize=*/true);
}

template <int BLOCK>
__global__ __launch_bounds__(BLOCK) void k_sigmoid_ep(void *in,
                                                      void *bias,
                                                      void *out,
                                                      void *routing,
                                                      void *active,
                                                      float scale,
                                                      int start_expert,
                                                      int end_expert) {
  kernel::topk_sigmoid_task_impl<bf16, /*VPT=*/8, /*EXPERTS=*/256,
                                 /*WARPS_PER_CTA=*/8, /*BYTES_PER_LDG=*/16,
                                 /*NUM_GROUPS=*/8, /*TOPK_GROUP=*/4,
                                 /*EXPERTS_PER_GROUP=*/32, /*TOPK_EXPERTS=*/8>(
      in, bias, /*finished=*/nullptr, out, /*num_rows=*/8, routing, active,
      start_expert, end_expert, scale);
}

template __global__ void
    k_softmax_ep<16, 256, 16, 256>(void *, void *, void *, void *, int, int, int);
template __global__ void
    k_softmax_ep<16, 256, 16, 128>(void *, void *, void *, void *, int, int, int);
template __global__ void
    k_softmax_ep<8, 128, 16, 64>(void *, void *, void *, void *, int, int, int);
template __global__ void
    k_sigmoid_ep<256>(void *, void *, void *, void *, void *, float, int, int);
template __global__ void
    k_sigmoid_ep<64>(void *, void *, void *, void *, void *, float, int, int);
