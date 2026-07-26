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

// Kernel-wrapper test harness for the Gated-DeltaNet causal conv1d task
// (include/mirage/persistent_kernel/tasks/blackwell/gdn_conv1d_sm100.cuh).
//
// One CUDA block per (REQUEST SLOT, CHANNEL BLOCK), mirroring the persistent
// runtime where the task graph emits grid (slots, channel blocks, 1) and each
// task is executed by one worker CTA. The wrapper does exactly the pointer
// arithmetic that `TaskRegister::register_gdn_conv1d_sm100_task` emits into the
// generated `_execute_task()`:
//
//   c0      = channel_block * CHANNELS
//   input  += qo_indptr[slot] * INPUT_STRIDE  + c0
//   weight += c0 * KERNEL_SIZE
//   output += qo_indptr[slot] * OUTPUT_STRIDE + c0
//   state  += slot * (KERNEL_SIZE - 1) * CONV_DIM + c0
//   q_len   = qo_indptr[slot + 1] - qo_indptr[slot]
//
// `zero_state` is supplied per slot instead of being derived from
// `runtime_config.step[request_ids[slot]]`, so a single launch can cover
// "fresh request" and "carried state" slots at once (that is the step==0
// predicate's only observable effect).

#include "blackwell/gdn_conv1d_sm100.cuh"
#include "runtime_header.h"
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

using bfloat16 = type::bfloat16_t;

namespace {

template <int CONV_DIM,
          int CHANNELS,
          int KERNEL_SIZE,
          int INPUT_STRIDE,
          int OUTPUT_STRIDE>
__global__ void __launch_bounds__(WORKER_NUM_THREADS)
    gdn_conv1d_wrapper(void const *input,
                       void const *weight,
                       void *state,
                       void *output,
                       int const *qo_indptr,
                       uint8_t const *zero_state) {
  int const slot = blockIdx.x;
  int const c0 = blockIdx.y * CHANNELS;
  int const tok0 = qo_indptr[slot];
  int const q_len = qo_indptr[slot + 1] - tok0;
  kernel::gdn_conv1d_sm100_task_impl<bfloat16,
                                     CONV_DIM,
                                     CHANNELS,
                                     KERNEL_SIZE,
                                     INPUT_STRIDE,
                                     OUTPUT_STRIDE>(
      static_cast<bfloat16 const *>(input) + (size_t)tok0 * INPUT_STRIDE + c0,
      static_cast<bfloat16 const *>(weight) + (size_t)c0 * KERNEL_SIZE,
      static_cast<bfloat16 *>(state) +
          (size_t)slot * (KERNEL_SIZE - 1) * CONV_DIM + c0,
      static_cast<bfloat16 *>(output) + (size_t)tok0 * OUTPUT_STRIDE + c0,
      q_len,
      zero_state[slot] != 0);
}

template <int CONV_DIM,
          int CHANNELS,
          int KERNEL_SIZE,
          int INPUT_STRIDE,
          int OUTPUT_STRIDE>
void launch_gdn_conv1d(int num_slots,
                       void const *input,
                       void const *weight,
                       void *state,
                       void *output,
                       int const *qo_indptr,
                       uint8_t const *zero_state) {
  dim3 grid(num_slots, CONV_DIM / CHANNELS, 1);
  gdn_conv1d_wrapper<CONV_DIM,
                     CHANNELS,
                     KERNEL_SIZE,
                     INPUT_STRIDE,
                     OUTPUT_STRIDE>
      <<<grid, WORKER_NUM_THREADS, 0, at::cuda::getCurrentCUDAStream()>>>(
          input, weight, state, output, qo_indptr, zero_state);
}

// Shapes this harness instantiates: (conv_dim, channels_per_task, kernel_size,
// input_stride, output_stride). Qwen3.5's GDN conv is CONV_DIM = 8192
// (2*key_dim + value_dim), KERNEL_SIZE = 4; the channel-block variants cover
// the prefill scaling axis (8192 -> 1/8/32 blocks), the small dims keep the
// unit tests fast, and the (8192, 12288) entry covers a strided input row,
// which is the layout a fused in_proj_qkvz would hand us
// (vllm-graph.md 2.1.7 row 6).
#define GDN_CONV1D_CASES(F)                                                    \
  F(8192, 8192, 4, 8192, 8192)                                                 \
  F(8192, 1024, 4, 8192, 8192)                                                 \
  F(8192, 256, 4, 8192, 8192)                                                  \
  F(8192, 8192, 4, 12288, 8192)                                                \
  F(8192, 256, 4, 12288, 8192)                                                 \
  F(4096, 4096, 4, 4096, 4096)                                                 \
  F(4096, 512, 4, 4096, 4096)                                                  \
  F(512, 512, 4, 512, 512)                                                     \
  F(512, 128, 4, 512, 512)                                                     \
  F(128, 128, 4, 128, 128)                                                     \
  F(128, 32, 4, 128, 128)                                                      \
  F(128, 128, 2, 128, 128)                                                     \
  F(128, 128, 8, 128, 128)

bool dispatch_gdn_conv1d(int conv_dim,
                         int channels_per_task,
                         int kernel_size,
                         int input_stride,
                         int output_stride,
                         int num_slots,
                         void const *input,
                         void const *weight,
                         void *state,
                         void *output,
                         int const *qo_indptr,
                         uint8_t const *zero_state) {
#define GDN_CONV1D_DISPATCH(D, C, K, IS, OS)                                   \
  if (conv_dim == (D) && channels_per_task == (C) && kernel_size == (K) &&     \
      input_stride == (IS) && output_stride == (OS)) {                         \
    launch_gdn_conv1d<D, C, K, IS, OS>(                                        \
        num_slots, input, weight, state, output, qo_indptr, zero_state);       \
    return true;                                                               \
  }
  GDN_CONV1D_CASES(GDN_CONV1D_DISPATCH)
#undef GDN_CONV1D_DISPATCH
  return false;
}

} // namespace

void gdn_conv1d_sm100(torch::Tensor input,      // [num_tokens, input_stride]
                      torch::Tensor weight,     // [conv_dim, kernel_size]
                      torch::Tensor state,      // [num_slots, k-1, conv_dim]
                      torch::Tensor output,     // [num_tokens, output_stride]
                      torch::Tensor qo_indptr,  // int32 [num_slots + 1]
                      torch::Tensor zero_state, // uint8 [num_slots]
                      int64_t num_channel_blocks) {
  TORCH_CHECK(input.dim() == 2 && input.is_contiguous() &&
                  input.scalar_type() == at::kBFloat16,
              "input must be a contiguous 2D bfloat16 tensor");
  TORCH_CHECK(output.dim() == 2 && output.is_contiguous() &&
                  output.scalar_type() == at::kBFloat16,
              "output must be a contiguous 2D bfloat16 tensor");
  TORCH_CHECK(weight.dim() == 2 && weight.is_contiguous() &&
                  weight.scalar_type() == at::kBFloat16,
              "weight must be a contiguous 2D bfloat16 tensor "
              "[conv_dim, kernel_size]");
  TORCH_CHECK(state.dim() == 3 && state.is_contiguous() &&
                  state.scalar_type() == at::kBFloat16,
              "state must be a contiguous 3D bfloat16 tensor "
              "[num_slots, kernel_size-1, conv_dim]");
  TORCH_CHECK(qo_indptr.dim() == 1 && qo_indptr.is_contiguous() &&
                  qo_indptr.scalar_type() == at::kInt,
              "qo_indptr must be a contiguous 1D int32 tensor");
  TORCH_CHECK(zero_state.dim() == 1 && zero_state.is_contiguous() &&
                  zero_state.scalar_type() == at::kByte,
              "zero_state must be a contiguous 1D uint8 tensor");

  int const conv_dim = (int)weight.size(0);
  int const kernel_size = (int)weight.size(1);
  int const num_slots = (int)state.size(0);
  int const input_stride = (int)input.size(1);
  int const output_stride = (int)output.size(1);

  TORCH_CHECK(state.size(1) == kernel_size - 1 && state.size(2) == conv_dim,
              "state must be [num_slots, kernel_size-1, conv_dim]");
  TORCH_CHECK(qo_indptr.size(0) == num_slots + 1,
              "qo_indptr must have num_slots + 1 entries");
  TORCH_CHECK(zero_state.size(0) == num_slots,
              "zero_state must have num_slots entries");
  TORCH_CHECK(input_stride >= conv_dim && output_stride >= conv_dim,
              "row stride cannot be smaller than conv_dim");
  TORCH_CHECK(input.size(0) == output.size(0),
              "input and output must have the same number of token rows");
  TORCH_CHECK(num_channel_blocks >= 1 &&
                  conv_dim % (int)num_channel_blocks == 0,
              "num_channel_blocks must divide conv_dim");
  int const channels_per_task = conv_dim / (int)num_channel_blocks;

  bool const dispatched = dispatch_gdn_conv1d(conv_dim,
                                              channels_per_task,
                                              kernel_size,
                                              input_stride,
                                              output_stride,
                                              num_slots,
                                              input.data_ptr(),
                                              weight.data_ptr(),
                                              state.data_ptr(),
                                              output.data_ptr(),
                                              qo_indptr.data_ptr<int>(),
                                              zero_state.data_ptr<uint8_t>());
  TORCH_CHECK(dispatched,
              "Unsupported gdn_conv1d_sm100 shape [conv_dim=",
              conv_dim,
              ", channels_per_task=",
              channels_per_task,
              ", kernel_size=",
              kernel_size,
              ", input_stride=",
              input_stride,
              ", output_stride=",
              output_stride,
              "]");
  C10_CUDA_CHECK(cudaGetLastError());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gdn_conv1d_sm100",
        &gdn_conv1d_sm100,
        "Gated-DeltaNet causal depthwise conv1d with a per-slot conv-state "
        "pool (SM100)",
        pybind11::arg("input"),
        pybind11::arg("weight"),
        pybind11::arg("state"),
        pybind11::arg("output"),
        pybind11::arg("qo_indptr"),
        pybind11::arg("zero_state"),
        pybind11::arg("num_channel_blocks") = 1);
}
