/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Kernel-wrapper test harness for the preserved-block-scale dense FP8 GEMM
// (include/mirage/persistent_kernel/tasks/blackwell/linear_fp8_blockscale_sm100
// .cuh). One CUDA block per launch, mirroring the persistent runtime where one
// task is executed by one worker CTA.

#include "blackwell/linear_fp8_blockscale_sm100.cuh"
#include "runtime_header.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <vector>

using bfloat16 = type::bfloat16_t;

namespace {

template <int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE, bool RESIDUAL>
__global__ void __launch_bounds__(WORKER_NUM_THREADS)
    linear_fp8_blockscale_wrapper(void const *input_fp8,
                                  void const *input_scale,
                                  void const *weight_fp8,
                                  void const *weight_scale,
                                  void const *residual,
                                  void *output) {
  kernel::linear_fp8_blockscale_task_impl<bfloat16,
                                          BATCH_SIZE,
                                          OUTPUT_SIZE,
                                          REDUCTION_SIZE,
                                          OUTPUT_SIZE,
                                          RESIDUAL>(
      input_fp8, input_scale, weight_fp8, weight_scale, residual, output);
}

template <int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE, bool RESIDUAL>
void launch_linear_fp8_blockscale(void const *input_fp8,
                                  void const *input_scale,
                                  void const *weight_fp8,
                                  void const *weight_scale,
                                  void const *residual,
                                  void *output) {
  // task_smem_bytes, not either path's own figure: the entry point dispatches
  // at compile time between the golden path and the ferret fast path, and the
  // two need different arenas.
  constexpr int smem_size = kernel::linear_fp8_blockscale::task_smem_bytes(
      BATCH_SIZE, REDUCTION_SIZE, OUTPUT_SIZE);
  auto *entry = linear_fp8_blockscale_wrapper<BATCH_SIZE,
                                              OUTPUT_SIZE,
                                              REDUCTION_SIZE,
                                              RESIDUAL>;
  cudaFuncSetAttribute(
      entry, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  entry<<<1, WORKER_NUM_THREADS, smem_size, at::cuda::getCurrentCUDAStream()>>>(
      input_fp8, input_scale, weight_fp8, weight_scale, residual, output);
}

// Shapes instantiated by this harness. The dense projections Qwen3.5 runs in
// FP8 are all K in {512, 2048, 4096} with a per-task N of 128; N = 256 and the
// larger batches exercise the multi-N-block and multi-M-tile loops.
#define FOR_EACH_BATCH_SIZE(F, ...)                                            \
  F(1, __VA_ARGS__)                                                            \
  F(2, __VA_ARGS__)                                                            \
  F(4, __VA_ARGS__)                                                            \
  F(8, __VA_ARGS__)                                                            \
  F(16, __VA_ARGS__)                                                           \
  F(64, __VA_ARGS__)                                                           \
  F(256, __VA_ARGS__)

#define FOR_EACH_OUTPUT_SIZE(F, ...)                                           \
  F(128, __VA_ARGS__)                                                          \
  F(256, __VA_ARGS__)

#define FOR_EACH_REDUCTION_SIZE(F, ...)                                        \
  F(512, __VA_ARGS__)                                                          \
  F(2048, __VA_ARGS__)                                                         \
  F(4096, __VA_ARGS__)

#define DISPATCH_REDUCTION(RED, BAT, OUT)                                      \
  if (reduction_size == RED) {                                                 \
    if (has_residual) {                                                        \
      launch_linear_fp8_blockscale<BAT, OUT, RED, true>(                       \
          input_fp8, input_scale, weight_fp8, weight_scale, residual, output); \
    } else {                                                                   \
      launch_linear_fp8_blockscale<BAT, OUT, RED, false>(                      \
          input_fp8, input_scale, weight_fp8, weight_scale, nullptr, output);  \
    }                                                                          \
    return true;                                                               \
  }

#define DISPATCH_OUTPUT(OUT, BAT)                                              \
  if (output_size == OUT) {                                                    \
    FOR_EACH_REDUCTION_SIZE(DISPATCH_REDUCTION, BAT, OUT)                      \
    return false;                                                              \
  }

#define DISPATCH_BATCH(BAT, UNUSED)                                            \
  if (batch_size == BAT) {                                                     \
    FOR_EACH_OUTPUT_SIZE(DISPATCH_OUTPUT, BAT)                                 \
    return false;                                                              \
  }

bool dispatch_linear_fp8_blockscale(int batch_size,
                                    int output_size,
                                    int reduction_size,
                                    bool has_residual,
                                    void const *input_fp8,
                                    void const *input_scale,
                                    void const *weight_fp8,
                                    void const *weight_scale,
                                    void const *residual,
                                    void *output) {
  FOR_EACH_BATCH_SIZE(DISPATCH_BATCH, 0)
  return false;
}

#undef DISPATCH_BATCH
#undef DISPATCH_OUTPUT
#undef DISPATCH_REDUCTION

// ---------------------------------------------------------------------------
// WHOLE-PROJECTION harness: N/N_SLICE cooperating tasks, exactly MPK's dispatch.
//
// This is the instrument for M4-I2's bit-exactness claim. The unit entry point
// above drives ONE task, which cannot express the sliced dispatch at all -- a
// sub-block slice only makes sense as part of a projection. Here the same
// projection is computed twice from identical inputs:
//
//   force_golden = true   N/128 tasks, `linear_fp8_blockscale_task_impl_golden`
//   force_golden = false  N/slice tasks, the dispatcher (ferret v011 fast path)
//
// and the test requires the two bf16 outputs to be BYTE-IDENTICAL. A sub-block
// slice lies inside one 128x128 scale block, so each task is handed its
// CONTAINING block row -- the same arithmetic MPK gets from the builder's
// row-replicated scale, expressed here as pointer math on the raw checkpoint
// layout so this harness also pins that indexing rule.
// ---------------------------------------------------------------------------
namespace {

template <int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int N_SLICE,
          bool RESIDUAL,
          bool FORCE_GOLDEN>
__global__ void __launch_bounds__(WORKER_NUM_THREADS)
    linear_fp8_blockscale_projection_kernel(uint8_t const *__restrict__ a_q,
                                            float const *__restrict__ a_s,
                                            uint8_t const *__restrict__ b_q,
                                            float const *__restrict__ b_s,
                                            bfloat16 const *__restrict__ res,
                                            bfloat16 *__restrict__ out) {
  int const nb = blockIdx.x;
  uint8_t const *w = b_q + (size_t)nb * N_SLICE * REDUCTION_SIZE;
  float const *ws =
      b_s + (size_t)(nb * N_SLICE / 128) * (REDUCTION_SIZE / 128);
  bfloat16 const *r = RESIDUAL ? res + (size_t)nb * N_SLICE : nullptr;
  bfloat16 *o = out + (size_t)nb * N_SLICE;
  if constexpr (FORCE_GOLDEN) {
    kernel::linear_fp8_blockscale_task_impl_golden<bfloat16,
                                                   BATCH_SIZE,
                                                   N_SLICE,
                                                   REDUCTION_SIZE,
                                                   OUTPUT_SIZE,
                                                   RESIDUAL>(
        a_q, a_s, w, ws, r, o);
  } else {
    kernel::linear_fp8_blockscale_task_impl<bfloat16,
                                            BATCH_SIZE,
                                            N_SLICE,
                                            REDUCTION_SIZE,
                                            OUTPUT_SIZE,
                                            RESIDUAL>(a_q, a_s, w, ws, r, o);
  }
}

template <int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int N_SLICE,
          bool RESIDUAL,
          bool FORCE_GOLDEN>
void launch_projection(void const *a_q,
                       void const *a_s,
                       void const *b_q,
                       void const *b_s,
                       void const *res,
                       void *out) {
  constexpr int smem_size =
      FORCE_GOLDEN
          ? kernel::linear_fp8_blockscale::smem_bytes(BATCH_SIZE)
          : kernel::linear_fp8_blockscale::task_smem_bytes(
                BATCH_SIZE, REDUCTION_SIZE, N_SLICE);
  auto *entry = linear_fp8_blockscale_projection_kernel<BATCH_SIZE,
                                                        OUTPUT_SIZE,
                                                        REDUCTION_SIZE,
                                                        N_SLICE,
                                                        RESIDUAL,
                                                        FORCE_GOLDEN>;
  cudaFuncSetAttribute(
      entry, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  entry<<<OUTPUT_SIZE / N_SLICE,
          WORKER_NUM_THREADS,
          smem_size,
          at::cuda::getCurrentCUDAStream()>>>(
      static_cast<uint8_t const *>(a_q),
      static_cast<float const *>(a_s),
      static_cast<uint8_t const *>(b_q),
      static_cast<float const *>(b_s),
      static_cast<bfloat16 const *>(res),
      static_cast<bfloat16 *>(out));
}

// The SHIPPED Qwen3.5 dense fp8 call sites: (N, K, WITH_RESIDUAL, per-shape
// N_SLICE). Slices are the ones the ferret winner (workspace4 v011) was
// benchmarked at and must stay in step with
// python/mirage/mpk/models/qwen3_5/builder.py's FP8_DENSE_N_SLICE.
#define FOR_EACH_SHIPPED_SHAPE(F, ...)                                         \
  F(8192, 2048, false, 64, __VA_ARGS__)  /* GDN in_proj_qkv    */              \
  F(4096, 2048, false, 32, __VA_ARGS__)  /* GDN in_proj_z      */              \
  F(9216, 2048, false, 64, __VA_ARGS__)  /* attn qkv(g)_proj   */              \
  F(2048, 4096, true, 16, __VA_ARGS__)   /* out_proj / o_proj  */              \
  F(1024, 2048, false, 32, __VA_ARGS__)  /* shared gate_up     */              \
  F(2048, 512, false, 64, __VA_ARGS__)   /* shared down        */

#define FOR_EACH_DECODE_BATCH(F, ...)                                          \
  F(1, __VA_ARGS__)                                                            \
  F(2, __VA_ARGS__)                                                            \
  F(4, __VA_ARGS__)                                                            \
  F(8, __VA_ARGS__)                                                            \
  F(16, __VA_ARGS__)

#define PROJ_SHAPE_CASE(N_, K_, RES_, SLICE_, BAT)                             \
  if (output_size == N_ && reduction_size == K_) {                             \
    if (force_golden) {                                                        \
      launch_projection<BAT, N_, K_, 128, RES_, true>(                         \
          a_q, a_s, b_q, b_s, res, out);                                       \
    } else {                                                                   \
      launch_projection<BAT, N_, K_, SLICE_, RES_, false>(                     \
          a_q, a_s, b_q, b_s, res, out);                                       \
    }                                                                          \
    return (RES_) == has_residual;                                             \
  }

#define PROJ_BATCH_CASE(BAT, UNUSED)                                           \
  if (batch_size == BAT) {                                                     \
    FOR_EACH_SHIPPED_SHAPE(PROJ_SHAPE_CASE, BAT)                               \
    return false;                                                              \
  }

bool dispatch_projection(int batch_size,
                         int output_size,
                         int reduction_size,
                         bool has_residual,
                         bool force_golden,
                         void const *a_q,
                         void const *a_s,
                         void const *b_q,
                         void const *b_s,
                         void const *res,
                         void *out) {
  FOR_EACH_DECODE_BATCH(PROJ_BATCH_CASE, 0)
  return false;
}

#undef PROJ_BATCH_CASE
#undef PROJ_SHAPE_CASE

} // namespace

void linear_fp8_blockscale_projection(torch::Tensor input_fp8,
                                     torch::Tensor input_scale,
                                     torch::Tensor weight_fp8,
                                     torch::Tensor weight_scale,
                                     c10::optional<at::Tensor> residual,
                                     torch::Tensor output,
                                     bool force_golden) {
  int const batch_size = static_cast<int>(input_fp8.size(0));
  int const reduction_size = static_cast<int>(input_fp8.size(1));
  int const output_size = static_cast<int>(weight_fp8.size(0));
  TORCH_CHECK(input_fp8.is_contiguous() && weight_fp8.is_contiguous() &&
                  input_scale.is_contiguous() &&
                  weight_scale.is_contiguous() && output.is_contiguous(),
              "all projection tensors must be contiguous");
  TORCH_CHECK(input_fp8.scalar_type() == at::kFloat8_e4m3fn &&
                  weight_fp8.scalar_type() == at::kFloat8_e4m3fn,
              "input_fp8/weight_fp8 must be float8_e4m3fn");
  TORCH_CHECK(input_scale.scalar_type() == at::kFloat &&
                  weight_scale.scalar_type() == at::kFloat,
              "block scales must stay float32 (no UE8M0 conversion)");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be bfloat16");
  TORCH_CHECK(weight_scale.size(0) == output_size / 128 &&
                  weight_scale.size(1) == reduction_size / 128,
              "weight_scale must be the checkpoint's RAW [N/128, K/128] "
              "float32 block scale (this harness does the containing-block "
              "indexing itself)");
  bool const has_residual = residual.has_value();
  bool const ok = dispatch_projection(
      batch_size,
      output_size,
      reduction_size,
      has_residual,
      force_golden,
      input_fp8.data_ptr(),
      input_scale.data_ptr(),
      weight_fp8.data_ptr(),
      weight_scale.data_ptr(),
      has_residual ? residual->data_ptr() : nullptr,
      output.data_ptr());
  TORCH_CHECK(ok,
              "Unsupported / residual-mismatched projection [B=",
              batch_size,
              ", N=",
              output_size,
              ", K=",
              reduction_size,
              ", residual=",
              has_residual,
              "]");
  C10_CUDA_CHECK(cudaGetLastError());
}

void linear_fp8_blockscale_sm100(torch::Tensor input_fp8,
                                 torch::Tensor input_scale,
                                 torch::Tensor weight_fp8,
                                 torch::Tensor weight_scale,
                                 c10::optional<at::Tensor> residual,
                                 torch::Tensor output) {
  TORCH_CHECK(input_fp8.dim() == 2 && input_fp8.is_contiguous(),
              "input_fp8 must be a contiguous 2D tensor");
  TORCH_CHECK(weight_fp8.dim() == 2 && weight_fp8.is_contiguous(),
              "weight_fp8 must be a contiguous 2D tensor");
  TORCH_CHECK(input_scale.dim() == 2 && input_scale.is_contiguous(),
              "input_scale must be a contiguous 2D tensor");
  TORCH_CHECK(weight_scale.dim() == 2 && weight_scale.is_contiguous(),
              "weight_scale must be a contiguous 2D tensor");
  TORCH_CHECK(output.dim() == 2 && output.is_contiguous(),
              "output must be a contiguous 2D tensor");
  TORCH_CHECK(input_fp8.scalar_type() == at::kFloat8_e4m3fn &&
                  weight_fp8.scalar_type() == at::kFloat8_e4m3fn,
              "input_fp8/weight_fp8 must be float8_e4m3fn");
  TORCH_CHECK(input_scale.scalar_type() == at::kFloat &&
                  weight_scale.scalar_type() == at::kFloat,
              "block scales must stay float32 (no UE8M0 conversion)");
  TORCH_CHECK(output.scalar_type() == at::kBFloat16, "output must be bfloat16");

  int const batch_size = static_cast<int>(input_fp8.size(0));
  int const reduction_size = static_cast<int>(input_fp8.size(1));
  int const output_size = static_cast<int>(weight_fp8.size(0));

  TORCH_CHECK(weight_fp8.size(1) == reduction_size,
              "weight_fp8 reduction dim mismatch");
  TORCH_CHECK(output.size(0) == batch_size && output.size(1) == output_size,
              "output shape mismatch");
  TORCH_CHECK(input_scale.size(0) == batch_size &&
                  input_scale.size(1) == reduction_size / 128,
              "input_scale must be [batch, K/128] float32");
  TORCH_CHECK(weight_scale.size(0) == output_size / 128 &&
                  weight_scale.size(1) == reduction_size / 128,
              "weight_scale must be the checkpoint's [N/128, K/128] float32 "
              "block scale");

  bool const has_residual = residual.has_value();
  if (has_residual) {
    TORCH_CHECK(residual->dim() == 2 && residual->is_contiguous() &&
                    residual->scalar_type() == at::kBFloat16,
                "residual must be a contiguous 2D bfloat16 tensor");
    TORCH_CHECK(residual->size(0) == batch_size &&
                    residual->size(1) == output_size,
                "residual shape mismatch");
  }

  bool const dispatched = dispatch_linear_fp8_blockscale(
      batch_size,
      output_size,
      reduction_size,
      has_residual,
      input_fp8.data_ptr(),
      input_scale.data_ptr(),
      weight_fp8.data_ptr(),
      weight_scale.data_ptr(),
      has_residual ? residual->data_ptr() : nullptr,
      output.data_ptr());
  TORCH_CHECK(dispatched,
              "Unsupported linear_fp8_blockscale_sm100 shape [B=",
              batch_size,
              ", N=",
              output_size,
              ", K=",
              reduction_size,
              "]");
  C10_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear_fp8_blockscale_sm100",
        &linear_fp8_blockscale_sm100,
        "Dense FP8 GEMM consuming the checkpoint's float32 128x128 block "
        "scales (SM100)");
  m.def("linear_fp8_blockscale_projection",
        &linear_fp8_blockscale_projection,
        "One whole dense FP8 projection as N/slice cooperating tasks, MPK's "
        "dispatch. force_golden=True pins the pre-M4-I2 golden path at the "
        "128-row slice, so the two calls form a bit-exactness A/B.",
        pybind11::arg("input_fp8"),
        pybind11::arg("input_scale"),
        pybind11::arg("weight_fp8"),
        pybind11::arg("weight_scale"),
        pybind11::arg("residual"),
        pybind11::arg("output"),
        pybind11::arg("force_golden") = false);
}
