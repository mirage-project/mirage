/* Copyright 2026 The Mirage Team. Licensed under the Apache License, Version 2.0. */
#pragma once
#include "tasks/common/common_header.cuh"
#include <math_constants.h>

namespace kernel {
// The offline scheduler advances step before each graph iteration. Keep the
// one-token probe's INT64 position and BF16 RoPE layout inside the graph.
template <typename T>
__device__ __forceinline__ void qwen38_runtime_position_task(
    void const *step_ptr, void const *rope_table_ptr,
    void *position_ptr, void *rope_ptr) {
  int step = *static_cast<int const *>(step_ptr);
  if (step < 0 || step >= 1024) return;
  if (threadIdx.x == 0)
    *static_cast<int64_t *>(position_ptr) = step;
  auto table = static_cast<T const *>(rope_table_ptr);
  auto rope = static_cast<T *>(rope_ptr);
  for (int i = threadIdx.x; i < 128; i += blockDim.x)
    rope[i] = table[step * 128 + i];
  __syncthreads();
}

template <typename T, int VOCAB, int HIDDEN>
__device__ __forceinline__ void qwen35_embed_task(
    void const *token, void const *weight, void *output) {
  auto id = *static_cast<int64_t const *>(token);
  auto w = static_cast<T const *>(weight);
  auto y = static_cast<T *>(output);
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x)
    y[i] = id >= 0 && id < VOCAB ? w[id * HIDDEN + i] : T(0.f);
  __syncthreads();
}

template <typename T, int HIDDEN>
__device__ __forceinline__ void qwen36_shared_gate_task(
    void const *gate_ptr, void const *shared_ptr, void *output_ptr) {
  auto gate = static_cast<T const *>(gate_ptr);
  auto shared = static_cast<T const *>(shared_ptr);
  auto output = static_cast<T *>(output_ptr);
  // HF applies sigmoid to the BF16 scalar, then multiplies the BF16 shared
  // expert result. Preserve the intermediate BF16 rounding boundary.
  T scale = T(1.f / (1.f + expf(-float(gate[0]))));
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x)
    output[i] = T(__fmul_rn(float(scale), float(shared[i])));
  __syncthreads();
}

// Full vocabulary, stable first-index tie break, no padded-vocabulary assumptions.
template <typename T, int VOCAB>
__device__ __forceinline__ void qwen35_argmax_task(void const *input, void *output) {
  auto x = static_cast<T const *>(input);
  __shared__ float values[256];
  __shared__ int indices[256];
  float best = -CUDART_INF_F;
  int index = VOCAB;
  for (int i = threadIdx.x; i < VOCAB; i += blockDim.x) {
    float v = float(x[i]);
    if (v > best || (v == best && i < index)) { best = v; index = i; }
  }
  values[threadIdx.x] = best;
  indices[threadIdx.x] = index;
  __syncthreads();
  for (int stride = 128; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      int j = threadIdx.x + stride;
      if (values[j] > values[threadIdx.x] ||
          (values[j] == values[threadIdx.x] && indices[j] < indices[threadIdx.x])) {
        values[threadIdx.x] = values[j]; indices[threadIdx.x] = indices[j];
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) *static_cast<int64_t *>(output) = indices[0];
  __syncthreads();
}
} // namespace kernel
