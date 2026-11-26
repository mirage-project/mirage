/* Copyright 2026 The Mirage Team. Licensed under the Apache License, Version 2.0. */
#pragma once
#include "tasks/common/common_header.cuh"

namespace kernel {

template <typename T, int BATCH, int TOPK, int HIDDEN, int EXPERTS>
__device__ __forceinline__ void qwen36_moe_sum_task(
    void const *expert_output_ptr, void const *weight_ptr,
    void const *routing_indices_ptr, void const *shared_ptr, void *output_ptr) {
  auto expert_output = static_cast<T const *>(expert_output_ptr);
  auto weight = static_cast<float const *>(weight_ptr);
  auto routing_indices = static_cast<int const *>(routing_indices_ptr);
  auto shared = static_cast<T const *>(shared_ptr);
  auto output = static_cast<T *>(output_ptr);
  // HF's default grouped_mm backend restores top-k slot order, rounds each
  // weighted expert output to BF16, then sums in FP32 before rounding once.
  // The separately gated shared expert is added at the BF16 boundary.
  for (int row = 0; row < BATCH; ++row) {
    int count = 0;
    for (int expert = 0; expert < EXPERTS; ++expert) {
      int slot = routing_indices[expert * BATCH + row] - 1;
      if (slot >= 0 && slot < TOPK) {
#ifndef MPK_QWEN36_ROUTING_DIAGNOSTIC
        if (count == TOPK) __trap();
#endif
        ++count;
      }
    }
#ifndef MPK_QWEN36_ROUTING_DIAGNOSTIC
    if (count != TOPK) __trap();
#endif
    for (int col = threadIdx.x; col < HIDDEN; col += blockDim.x) {
      // torch.sum(dim=1) uses Reduce.cuh's four per-thread accumulators for
      // this [BATCH, TOPK, HIDDEN] layout: (0+4), ..., (3+7), then 0..3.
      float routed_parts[4] = {0.f, 0.f, 0.f, 0.f};
      for (int slot = 0; slot < TOPK; ++slot) {
        T contribution = T(float(expert_output[(row * TOPK + slot) * HIDDEN + col]) *
                           float(T(weight[row * TOPK + slot])));
        routed_parts[slot % 4] += float(contribution);
      }
      float routed = ((routed_parts[0] + routed_parts[1]) + routed_parts[2]) +
                     routed_parts[3];
      output[row * HIDDEN + col] =
          T(float(T(routed)) + float(shared[row * HIDDEN + col]));
    }
  }
  __syncthreads();
}

} // namespace kernel
