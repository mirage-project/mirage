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

// Standalone drivers for the NON-GEMM tasks of the Qwen3.5 MoE block (M2-I7).
//
// The grouped FP8 GEMMs already have a harness
// (sm100_fp8_moe_qwen35/runtime_kernel_wrapper_qwen35.cu, M2-I13). This one
// covers what surrounds them:
//
//   * topk_softmax_sm100  -- the router. Probe P5 needs to drive it with an
//     EXPLICIT `VPT` rather than the `TopkConstants` default, because VPT is
//     what sets the kernel's rows-per-PASS
//     (ROWS_PER_WARP = WARP_SIZE * VPT / NUM_EXPERTS, 8 warps per block), and
//     P5's job was to establish whether the shipped instantiation covers our
//     `mbt = 16` build (docs/qwen35/v1-architecture.md 9.1). Since M3-I5b the
//     kernel loops over row tiles, so both VPTs must agree on every row and
//     `vpt` selects how many passes it takes, not how many rows are routed.
//   * sigmoid_gate_mul_add_sm100 (id 238) -- the new shared-expert gate task.
//   * mul_sum_add_sm100 -- the combine, so the router weight -> combine
//     boundary can be checked on the same bytes.
//
// Every driver launches ONE thread block, exactly like a megakernel worker
// executing one task.

#include "blackwell/task_header.cuh"
#include "runtime_header.h"
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <cstdio>

using bfloat16 = cute::bfloat16_t;

namespace {

// ================================================================
// topk_softmax_sm100 -- the router (existing kernel, id 260)
// ================================================================
template <typename T, int VPT, int EXPERTS, int BYTES_PER_LDG>
__global__ __launch_bounds__(256) void q35_topk_softmax_kernel(
    void *__restrict__ gating_output,
    void *__restrict__ topk_weights,
    void *__restrict__ mpk_routing_indices,
    void *__restrict__ mpk_active_expert_ids,
    int num_rows,
    int k,
    bool renormalize,
    bool round_weights) {
  kernel::topk_softmax_task_impl<T,
                                 VPT,
                                 EXPERTS,
                                 /*WARPS_PER_CTA=*/8,
                                 BYTES_PER_LDG>(gating_output,
                                                /*finished=*/nullptr,
                                                topk_weights,
                                                num_rows,
                                                k,
                                                mpk_routing_indices,
                                                mpk_active_expert_ids,
                                                /*start_expert=*/0,
                                                /*end_expert=*/EXPERTS,
                                                renormalize,
                                                round_weights);
  __syncthreads();
}

// ================================================================
// sigmoid_gate_mul_add_sm100 -- the new shared-expert gate (id 238)
// ================================================================
template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int HIDDEN_SIZE>
__global__ __launch_bounds__(256) void q35_sigmoid_gate_mul_add_kernel(
    void const *x_ptr,
    void const *gate_w_ptr,
    void const *shared_ptr,
    void const *residual_ptr,
    void *output_ptr) {
  kernel::sigmoid_gate_mul_add_task_impl<T,
                                         BATCH_SIZE,
                                         OUTPUT_SIZE,
                                         HIDDEN_SIZE,
                                         /*X_STRIDE=*/HIDDEN_SIZE,
                                         /*O_STRIDE=*/OUTPUT_SIZE>(
      x_ptr, gate_w_ptr, shared_ptr, residual_ptr, output_ptr);
  __syncthreads();
}

// ================================================================
// mul_sum_add_sm100 -- the combine (existing kernel, id 261)
// ================================================================
template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int NUM_TOPK>
__global__
    __launch_bounds__(256) void q35_mul_sum_add_kernel(void const *input_ptr,
                                                       void const *weight_ptr,
                                                       void const *residual_ptr,
                                                       void *output_ptr) {
  kernel::mul_sum_add_sm100_task_impl<T,
                                      BATCH_SIZE,
                                      OUTPUT_SIZE,
                                      NUM_TOPK,
                                      /*OUTPUT_STRIDE=*/OUTPUT_SIZE>(
      input_ptr, weight_ptr, residual_ptr, output_ptr);
  __syncthreads();
}

void check_cuda(char const *what) {
  cudaError_t err = cudaDeviceSynchronize();
  if (err == cudaSuccess) {
    err = cudaGetLastError();
  }
  TORCH_CHECK(err == cudaSuccess, what, ": ", cudaGetErrorString(err));
}

} // namespace

// `vpt` selects the instantiation under test: 8 is what
// register_moe_topk_softmax_sm100_task ships today, 16 is the alternative that
// doubles the per-task row capacity. 0 means "whatever TopkConstants picks".
void topk_softmax_sm100(torch::Tensor gating_output,
                        torch::Tensor topk_weights,
                        torch::Tensor mpk_routing_indices,
                        torch::Tensor mpk_active_expert_ids,
                        int64_t vpt,
                        bool round_weights) {
  int const num_rows = static_cast<int>(gating_output.size(0));
  int const num_experts = static_cast<int>(gating_output.size(1));
  int const num_topk = static_cast<int>(topk_weights.size(1));
  TORCH_CHECK(gating_output.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(topk_weights.scalar_type() == torch::kFloat32);
  TORCH_CHECK(mpk_routing_indices.size(0) == num_experts &&
              mpk_routing_indices.size(1) == num_rows);
  TORCH_CHECK(mpk_active_expert_ids.size(0) == num_experts + 1);
  TORCH_CHECK(num_experts == 256, "P5 harness is instantiated for 256 experts");

  void *g = gating_output.data_ptr();
  void *w = topk_weights.data_ptr();
  void *r = mpk_routing_indices.data_ptr();
  void *a = mpk_active_expert_ids.data_ptr();
  dim3 grid(1, 1, 1), block(256, 1, 1);

  if (vpt == 0) {
    vpt = kernel::detail::TopkConstants<bfloat16, 256, 16>::VPT;
  }
  if (vpt == 8) {
    q35_topk_softmax_kernel<bfloat16, 8, 256, 16>
        <<<grid, block>>>(g, w, r, a, num_rows, num_topk, true, round_weights);
  } else if (vpt == 16) {
    q35_topk_softmax_kernel<bfloat16, 16, 256, 16>
        <<<grid, block>>>(g, w, r, a, num_rows, num_topk, true, round_weights);
  } else {
    TORCH_CHECK(false, "unsupported vpt=", vpt, " (expected 0, 8 or 16)");
  }
  check_cuda("topk_softmax_sm100");
}

// Reports the compile-time rows-per-PASS of an instantiation, so the probe can
// state it as a measured fact rather than a hand-derivation. Since M3-I5b this
// is a cost unit, not a capacity: the kernel repeats the pass until every row
// is routed.
int64_t topk_softmax_rows_per_task(int64_t vpt) {
  if (vpt == 0) {
    vpt = kernel::detail::TopkConstants<bfloat16, 256, 16>::VPT;
  }
  // ROWS_PER_WARP = (WARP_SIZE * VPT) / NUM_EXPERTS, over 8 warps.
  int64_t rows_per_warp = (32 * vpt) / 256;
  return rows_per_warp * 8;
}

int64_t topk_softmax_default_vpt() {
  return kernel::detail::TopkConstants<bfloat16, 256, 16>::VPT;
}

void sigmoid_gate_mul_add_sm100(torch::Tensor x,
                                torch::Tensor gate_weight,
                                torch::Tensor shared,
                                torch::Tensor residual,
                                torch::Tensor output) {
  int const batch = static_cast<int>(x.size(0));
  int const hidden = static_cast<int>(x.size(1));
  int const out_size = static_cast<int>(output.size(1));
  TORCH_CHECK(x.scalar_type() == torch::kBFloat16);
  TORCH_CHECK(gate_weight.size(0) == 1 && gate_weight.size(1) == hidden);
  TORCH_CHECK(shared.size(0) == batch && shared.size(1) == out_size);
  TORCH_CHECK(residual.size(0) == batch && residual.size(1) == out_size);
  TORCH_CHECK(output.size(0) == batch);

  void const *xp = x.data_ptr();
  void const *wp = gate_weight.data_ptr();
  void const *sp = shared.data_ptr();
  void const *rp = residual.data_ptr();
  void *op = output.data_ptr();
  // 9 floats: one per warp (<=8) plus the broadcast gate slot.
  size_t smem = 16 * sizeof(float);
  dim3 grid(1, 1, 1), block(256, 1, 1);

#define Q35_GATE_CASE(B, N, H)                                                 \
  if (batch == (B) && out_size == (N) && hidden == (H)) {                      \
    q35_sigmoid_gate_mul_add_kernel<bfloat16, B, N, H>                         \
        <<<grid, block, smem>>>(xp, wp, sp, rp, op);                           \
    check_cuda("sigmoid_gate_mul_add_sm100");                                  \
    return;                                                                    \
  }
  // Qwen3.5 decode / prefill-chunk shapes (hidden = out = 2048).
  Q35_GATE_CASE(1, 2048, 2048)
  Q35_GATE_CASE(2, 2048, 2048)
  Q35_GATE_CASE(8, 2048, 2048)
  Q35_GATE_CASE(16, 2048, 2048)
  // Small shapes for fast unit sweeps.
  Q35_GATE_CASE(1, 256, 256)
  Q35_GATE_CASE(4, 256, 256)
  Q35_GATE_CASE(3, 512, 256)
#undef Q35_GATE_CASE
  TORCH_CHECK(false,
              "unsupported (batch, out, hidden) = (",
              batch,
              ", ",
              out_size,
              ", ",
              hidden,
              ") -- add a Q35_GATE_CASE");
}

void mul_sum_add_sm100(torch::Tensor input,
                       torch::Tensor weight,
                       torch::Tensor residual,
                       torch::Tensor output) {
  int const batch = static_cast<int>(output.size(0));
  int const out_size = static_cast<int>(output.size(1));
  int const num_topk = static_cast<int>(input.size(1));
  TORCH_CHECK(weight.scalar_type() == torch::kFloat32);
  TORCH_CHECK(input.size(0) == batch && input.size(2) == out_size);
  TORCH_CHECK(residual.size(0) == batch && residual.size(1) == out_size);

  void const *ip = input.data_ptr();
  void const *wp = weight.data_ptr();
  void const *rp = residual.data_ptr();
  void *op = output.data_ptr();
  dim3 grid(1, 1, 1), block(256, 1, 1);

#define Q35_COMBINE_CASE(B, N, K)                                              \
  if (batch == (B) && out_size == (N) && num_topk == (K)) {                    \
    q35_mul_sum_add_kernel<bfloat16, B, N, K>                                  \
        <<<grid, block>>>(ip, wp, rp, op);                                     \
    check_cuda("mul_sum_add_sm100");                                           \
    return;                                                                    \
  }
  Q35_COMBINE_CASE(1, 2048, 8)
  Q35_COMBINE_CASE(2, 2048, 8)
  Q35_COMBINE_CASE(8, 2048, 8)
  Q35_COMBINE_CASE(16, 2048, 8)
#undef Q35_COMBINE_CASE
  TORCH_CHECK(false,
              "unsupported (batch, out, topk) = (",
              batch,
              ", ",
              out_size,
              ", ",
              num_topk,
              ") -- add a Q35_COMBINE_CASE");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("topk_softmax_sm100",
        &topk_softmax_sm100,
        "MoE router: fp32 softmax over all experts -> top-k -> renormalize",
        pybind11::arg("gating_output"),
        pybind11::arg("topk_weights"),
        pybind11::arg("mpk_routing_indices"),
        pybind11::arg("mpk_active_expert_ids"),
        pybind11::arg("vpt") = 0,
        pybind11::arg("round_weights") = false);
  m.def("topk_softmax_rows_per_task",
        &topk_softmax_rows_per_task,
        "Rows one topk_softmax task can process at a given VPT");
  m.def("topk_softmax_default_vpt",
        &topk_softmax_default_vpt,
        "VPT that TopkConstants picks for 256 bf16 experts");
  m.def("sigmoid_gate_mul_add_sm100",
        &sigmoid_gate_mul_add_sm100,
        "Qwen3.5 shared-expert gate: residual + sigmoid(x.w_sg) * shared");
  m.def("mul_sum_add_sm100",
        &mul_sum_add_sm100,
        "MoE combine: sum_j w_j * y_j + residual");
}
