/* Copyright 2026 The Mirage Team. Licensed under the Apache License, Version 2.0. */
#pragma once
#include "tasks/common/common_header.cuh"
#include <type_traits>

// Correctness-first, one-token Qwen3.8-27B tasks. State is private to one
// request. The test-mode caller synchronizes launches before reusing/resetting
// it. These tasks do not implement the serving scheduler's state lifecycle.
namespace kernel {

// Finalize the 16 accumulators used by PyTorch's strided FP32 sum of a
// [16,128,128] state over its key axis: four accumulators per warp and four
// cooperating warps (Reduce.cuh, output_vec_size=4, block=(32,4), vt0=4).
__device__ __forceinline__ float qwen35_sum_state_partials(float const *s) {
  float warps[4];
  #pragma unroll
  for (int y = 0; y < 4; ++y)
    warps[y] = __fadd_rn(__fadd_rn(__fadd_rn(s[y], s[4+y]), s[8+y]), s[12+y]);
  return __fadd_rn(__fadd_rn(warps[0], warps[2]), __fadd_rn(warps[1], warps[3]));
}

// Pinned PyTorch CUDA reduction for 16 contiguous rows of width 128:
// Reduce.cuh selects block=(32,16), without input vectorization (width <=128).
// A lane combines x[i], x[i+32], x[i+64], x[i+96] in that order before the
// ascending warp reduction. A 128-lane pairwise tree changes FP32 RMS means
// enough to cross BF16 rounding boundaries. All 256 task threads participate.
__device__ __forceinline__ float qwen35_sum_128(float value) {
  __shared__ float sums[128];
  if (threadIdx.x < 128) sums[threadIdx.x] = value;
  __syncthreads();
  if (threadIdx.x < 32) {
    for (int offset = 32; offset < 128; offset += 32)
      value = __fadd_rn(value, sums[threadIdx.x + offset]);
    for (int offset = 1; offset < 32; offset <<= 1)
      value = __fadd_rn(value, __shfl_down_sync(0xffffffff, value, offset));
    if (threadIdx.x == 0) sums[0] = value;
  }
  __syncthreads();
  float result = sums[0];
  __syncthreads();
  return result;
}

// PyTorch Reduce.cuh uses up to 512 threads, sharing them across the
// power-of-two number of output rows (capped at 16). Each lane loads four
// adjacent FP32 squares. Batch size therefore changes the reduction order.
__host__ __device__ constexpr int qwen35_norm_lanes(int width, int rows) {
  int height = 1;
  while (height * 2 <= rows && height < 16) height *= 2;
  int lanes = 512 / height;
  while (lanes > width / 4) lanes /= 2;
  return lanes;
}

template <typename T, int HIDDEN, int ROWS=1>
__device__ __forceinline__ void qwen35_norm_task(
    void const *input, void const *weight, void *output) {
  auto x = static_cast<T const *>(input);
  auto w = static_cast<float const *>(weight);
  auto y = static_cast<T *>(output);
  __shared__ float inv_rms;
  if constexpr (HIDDEN == 1024 || HIDDEN == 2048 || HIDDEN == 5120) {
    constexpr int LANES = qwen35_norm_lanes(HIDDEN, ROWS);
    __shared__ float sums[LANES];
    for (int lane = threadIdx.x; lane < LANES; lane += blockDim.x) {
      float parts[4] = {};
      for (int at = lane * 4; at < HIDDEN; at += LANES * 4)
        for (int j = 0; j < 4; ++j) {
          float v = float(x[at+j]);
          parts[j] = __fadd_rn(parts[j], __fmul_rn(v, v));
        }
      sums[lane] = __fadd_rn(__fadd_rn(__fadd_rn(parts[0], parts[1]), parts[2]), parts[3]);
    }
    __syncthreads();
    float ss = threadIdx.x < LANES ? sums[threadIdx.x] : 0.f;
    for (int offset = LANES / 2; offset >= 32; offset >>= 1) {
      __syncthreads();
      if (threadIdx.x < offset) {
        ss = __fadd_rn(ss, sums[threadIdx.x+offset]);
        sums[threadIdx.x] = ss;
      }
    }
    __syncthreads();
    if (threadIdx.x < 32) {
      for (int offset = 1; offset < 32; offset <<= 1)
        ss = __fadd_rn(ss, __shfl_down_sync(0xffffffff, ss, offset));
      if (threadIdx.x == 0) inv_rms = rsqrtf(__fadd_rn(__fmul_rn(ss, 1.f / HIDDEN), 1.e-6f));
    }
  } else if (threadIdx.x == 0) {
    float ss = 0.f;
    for (int i = 0; i < HIDDEN; ++i)
      ss = __fadd_rn(ss, __fmul_rn(float(x[i]), float(x[i])));
    inv_rms = rsqrtf(ss / HIDDEN + 1.e-6f);
  }
  __syncthreads();
  for (int i = threadIdx.x; i < HIDDEN; i += blockDim.x)
    y[i] = T(__fmul_rn(__fmul_rn(float(x[i]), inv_rms), 1.f + w[i]));
  __syncthreads();
}

template <typename T, int CHANNELS>
__device__ __forceinline__ void qwen35_conv_task(
    void const *input, void const *weight, void const *history, void *output) {
  auto x = static_cast<T const *>(input);
  auto w = static_cast<T const *>(weight);
  auto s = const_cast<T *>(static_cast<T const *>(history));
  auto y = static_cast<T *>(output);
  for (int c = threadIdx.x; c < CHANNELS; c += blockDim.x) {
    float acc = 0.f;
    for (int j = 0; j < 3; ++j)
      acc = __fadd_rn(acc, __fmul_rn(float(s[c * 3 + j]), float(w[c * 4 + j])));
    acc = __fadd_rn(acc, __fmul_rn(float(x[c]), float(w[c * 4 + 3])));
    // conv1d returns BF16 before the separate SiLU operation.
    float v = float(T(acc));
    y[c] = T(v / (1.f + expf(-v)));
    s[c * 3] = s[c * 3 + 1];
    s[c * 3 + 1] = s[c * 3 + 2];
    s[c * 3 + 2] = x[c];
  }
  __syncthreads();
}

// Shared by the recurrent task and standalone numerical tests. Keep the
// checkpoint's norm-weight dtype and the BF16 boundary before weighting.
template <typename T, typename NormT, int D>
__device__ __forceinline__ void qwen35_norm_gate_task(
    void const *raw_output, void const *z, void const *norm_weight, void *output) {
  auto raw = static_cast<T const *>(raw_output);
  auto gate = static_cast<T const *>(z);
  auto w = static_cast<NormT const *>(norm_weight);
  auto y = static_cast<T *>(output);
  __shared__ float inv_rms;
  static_assert(D == 128);
  float value = threadIdx.x < D ? float(raw[threadIdx.x]) : 0.f;
  float ss = qwen35_sum_128(__fmul_rn(value, value));
  if (threadIdx.x == 0)
    inv_rms = rsqrtf(__fadd_rn(__fmul_rn(ss, 1.f / D), 1.e-6f));
  __syncthreads();
  if (threadIdx.x < D) {
    int i = threadIdx.x;
    float normalized = float(T(__fmul_rn(value, inv_rms)));
    float weighted = __fmul_rn(normalized, float(w[i]));
    if constexpr (std::is_same_v<NormT, T>) weighted = float(T(weighted));
    float g = float(gate[i]);
    y[i] = T(weighted * (g / (1.f + expf(-g))));
  }
  __syncthreads();
}

template <typename T, typename NormT, int HEADS, int D, int KEY_HEADS>
__device__ __forceinline__ void qwen35_delta_task(
    void const *qkv, void const *z, void const *ab, void const *parameters,
    void const *norm_weight, void const *state, void const *position,
    void *raw_output, void *output, void const *head_id=nullptr) {
  // Each task owns one head. Runtime imaps have already offset all pointers
  // except the shared norm weight. QKV rows and A/B rows retain full strides.
  auto x = static_cast<T const *>(qkv);
  // The grouped variant receives the full [Q(KH), K(KH), V(VH)] buffer.
  // Other inputs are already partitioned by value head by the MPK runtime.
  int vh = head_id ? int(*static_cast<int64_t const *>(head_id)) : 0;
  int kh = vh / (HEADS / KEY_HEADS);
  auto qx = x + kh * D;
  auto kx = x + KEY_HEADS * D + kh * D;
  auto vx = x + 2 * KEY_HEADS * D + vh * D;
  auto abv = static_cast<T const *>(ab);
  auto p = static_cast<float const *>(parameters); // [A_log; dt_bias]
  auto s = const_cast<float *>(static_cast<float const *>(state));
  int64_t pos = *static_cast<int64_t const *>(position);
  if (pos < 0 || pos >= 1024) return;
  auto raw = static_cast<T *>(raw_output);
  __shared__ float q[D], k[D], q_inv, k_inv, decay, beta, initial_qk;
  static_assert(D == 128);
  float qsq = 0.f, ksq = 0.f;
  if (threadIdx.x < D) {
    float qv = float(qx[threadIdx.x]), kv = float(kx[threadIdx.x]);
    qsq = float(T(__fmul_rn(qv, qv)));
    ksq = float(T(__fmul_rn(kv, kv)));
  }
  float qs = qwen35_sum_128(qsq), ks = qwen35_sum_128(ksq);
  if (threadIdx.x == 0) {
    // HF 5.14.1 performs Q/K L2 normalization before widening to FP32.
    q_inv = float(T(rsqrtf(float(T(float(T(qs)) + 1.e-6f)))));
    k_inv = float(T(rsqrtf(float(T(float(T(ks)) + 1.e-6f)))));
    float dt = float(abv[0]) + p[HEADS];
    float softplus = dt > 20.f ? dt : log1pf(expf(dt));
    decay = expf(-expf(p[0]) * softplus);
    beta = float(T(1.f / (1.f + expf(-float(abv[HEADS])))));
  }
  __syncthreads();
  if (threadIdx.x < D) {
    int i = threadIdx.x;
    q[i] = float(T(__fmul_rn(float(qx[i]), q_inv))) * rsqrtf(float(D));
    k[i] = float(T(__fmul_rn(float(kx[i]), k_inv)));
  }
  __syncthreads();
  // With one valid token and an empty cache, HF's padded chunk algorithm
  // reduces to (q @ k.T) * (beta * v), with state k.T @ (beta * v).
  // Reading q from the newly rounded state is mathematically equivalent but
  // introduces a different FP32 rounding boundary before the BF16 output.
  if (pos == 0 && threadIdx.x == 0) {
    float dot = 0.f;
    for (int i = 0; i < D; ++i) dot = __fmaf_rn(q[i], k[i], dot);
    initial_qk = dot;
  }
  __syncthreads();
  if (pos == 0 && threadIdx.x < D) {
    int v = threadIdx.x;
    float correction = __fmul_rn(float(vx[v]), beta);
    for (int i = 0; i < D; ++i)
      s[i * D + v] = __fmul_rn(k[i], correction);
    float value = __fmul_rn(initial_qk, correction);
    raw[v] = T(value);
#ifdef MPK_QWEN38_GDN_DUAL_LANE
  } else if (pos > 0) {
    // Two threads own each value column. Each keeps the same eight FP32
    // accumulators it would have owned in the 16-lane PyTorch reduction;
    // thread v combines the halves in the original order after each pass.
    int v = threadIdx.x % D;
    int half = threadIdx.x / D;
    extern __shared__ float qwen38_gdn_scratch[];
#ifdef MPK_QWEN38_B8_GDN_CACHE_SMEM
    float *cached = qwen38_gdn_scratch;
    float *upper = cached + D * D;
#else
    float *upper = qwen38_gdn_scratch;
#endif
    float *corrections = upper + 8 * D;
    float partials[8] = {};
    for (int base = 0; base < D; base += 16) {
      for (int slot = 0; slot < 8; ++slot) {
        int i = base + 8 * half + slot;
        float decayed = __fmul_rn(s[i * D + v], decay);
#ifdef MPK_QWEN38_B8_GDN_CACHE_SMEM
        cached[i * D + v] = decayed;
#endif
        partials[slot] = __fadd_rn(partials[slot], __fmul_rn(decayed, k[i]));
      }
    }
    if (half)
      for (int slot = 0; slot < 8; ++slot)
        upper[slot * D + v] = partials[slot];
    __syncthreads();
    if (!half) {
      float ordered[16];
      for (int slot = 0; slot < 8; ++slot) {
        ordered[slot] = partials[slot];
        ordered[8 + slot] = upper[slot * D + v];
      }
      float prediction = qwen35_sum_state_partials(ordered);
      corrections[v] = __fmul_rn(float(vx[v]) - prediction, beta);
    }
    __syncthreads();
    float correction = corrections[v];
    float readout[8] = {};
    for (int base = 0; base < D; base += 16) {
      for (int slot = 0; slot < 8; ++slot) {
        int i = base + 8 * half + slot;
#ifdef MPK_QWEN38_B8_GDN_CACHE_SMEM
        float decayed = cached[i * D + v];
#else
        float decayed = __fmul_rn(s[i * D + v], decay);
#endif
        float updated = __fadd_rn(decayed, __fmul_rn(k[i], correction));
        s[i * D + v] = updated;
        readout[slot] = __fadd_rn(readout[slot], __fmul_rn(updated, q[i]));
      }
    }
    if (half)
      for (int slot = 0; slot < 8; ++slot)
        upper[slot * D + v] = readout[slot];
    __syncthreads();
    if (!half) {
      float ordered[16];
      for (int slot = 0; slot < 8; ++slot) {
        ordered[slot] = readout[slot];
        ordered[8 + slot] = upper[slot * D + v];
      }
      raw[v] = T(qwen35_sum_state_partials(ordered));
    }
#else
  } else if (threadIdx.x < D) {
    int v = threadIdx.x;
    float prediction_sums[16] = {};
#ifdef MPK_QWEN38_B8_GDN_CACHE_SMEM
    // The mixed-task worker already reserves dynamic shared memory for other
    // task types. Reuse its first 64 KiB while this GDN task owns the CTA.
    // Keep the FP32 reduction and update order unchanged.
    extern __shared__ float qwen38_gdn_cached_state[];
#endif
#ifdef MPK_QWEN38_B8_GDN_UNROLL4
#pragma unroll 4
#endif
    for (int i = 0; i < D; ++i) {
      float decayed = __fmul_rn(s[i * D + v], decay);
#ifdef MPK_QWEN38_B8_GDN_CACHE_SMEM
      qwen38_gdn_cached_state[i * D + v] = decayed;
#endif
      prediction_sums[i % 16] = __fadd_rn(prediction_sums[i % 16], __fmul_rn(decayed, k[i]));
    }
    float prediction = qwen35_sum_state_partials(prediction_sums);
    float correction = __fmul_rn(float(vx[v]) - prediction, beta);
    float readout_sums[16] = {};
#ifdef MPK_QWEN38_B8_GDN_UNROLL4
#pragma unroll 4
#endif
    for (int i = 0; i < D; ++i) {
#ifdef MPK_QWEN38_B8_GDN_CACHE_SMEM
      float decayed = qwen38_gdn_cached_state[i * D + v];
#else
      float decayed = __fmul_rn(s[i * D + v], decay);
#endif
      float updated = __fadd_rn(decayed,
                               __fmul_rn(k[i], correction));
      s[i * D + v] = updated;
      readout_sums[i % 16] = __fadd_rn(readout_sums[i % 16], __fmul_rn(updated, q[i]));
    }
    float value = qwen35_sum_state_partials(readout_sums);
    raw[v] = T(value);
#endif
  }
  __syncthreads();
  qwen35_norm_gate_task<T, NormT, D>(raw_output, z, norm_weight, output);
}
template <typename T, typename NormT, int HEADS, int KEY_HEADS>
__device__ __forceinline__ void qwen35_delta_grouped_task(
    void const *qkv, void const *z, void const *ab, void const *parameters,
    void const *norm_weight, void const *state, void const *position,
    void const *head_id, void *raw_output, void *output) {
  qwen35_delta_task<T, NormT, HEADS, 128, KEY_HEADS>(
      qkv, z, ab, parameters, norm_weight, state, position, raw_output, output, head_id);
}
} // namespace kernel
