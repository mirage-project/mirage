// tasks/sm90_common.cuh -- what the sm90_* task types share: each runs an MPK
// Hopper task body (include/mirage/persistent_kernel/tasks/hopper/) as a task
// of the static megakernel. The body reads and writes its tensors by pointer
// (graph tensors: the node's buffer slots) and takes the dynamic shared memory
// from its start, over the ring.
#pragma once
#include "../core.cuh"

namespace static_mk {

// Before the body: every producer node of IN... (a graph input's StaticNode<0,
// 0, 0> has x == 0) has counted all its tasks, and this SM's earlier tasks are
// done with the shared memory (the ring's jobs end with their last warp, so the
// first barrier waits for them too).
template <class... IN>
__device__ __forceinline__ void sm90_task_begin(G const &g) {
  __syncthreads();
  if (threadIdx.x == 0) {
    (
        [&] {
          if constexpr (IN::x > 0 && IN::counter >= 0) {
            cnt_wait(g.cnt + IN::counter, IN::x * IN::y * IN::z);
          }
        }(),
        ...);
  }
  __syncthreads();
}

// After the body: its outputs are written; count the task in its node's
// counter (release: the CTA's stores, ordered by the barrier, before it).
template <class SELF>
__device__ __forceinline__ void sm90_task_end(G const &g) {
  __syncthreads();
  if constexpr (SELF::counter >= 0) {
    if (threadIdx.x == 0) {
      cnt_add(g.cnt + SELF::counter, 1);
    }
  }
}

// the dynamic shared memory an MPK Hopper body may take
constexpr int SM90_BODY_SMEM = 200 * 1024;

} // namespace static_mk
