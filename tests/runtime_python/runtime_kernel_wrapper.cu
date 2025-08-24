#include "argmax.cuh"
#include "bfloat16.h"
#include "linear.cuh"
#include "multitoken_paged_attention.cuh"
#include "norm.cuh"
#include "norm_linear.cuh"
// #include "norm_linear_original.cuh"
#include "bfloat16.h"
#include "embedding.cuh"
#include "paged_attention.cuh"
#include "prompt_lookup.cuh"
#include "silu_mul_linear.cuh"
#include "single_batch_decoding.cuh"
#include "single_batch_extend.cuh"
#include "single_batch_gqa.cuh"
#include "target_verify.cuh"
#include <cstdio>
#include <cuda_runtime.h>
#include <torch/extension.h>

// using kernel::argmax_kernel;
using kernel::argmax_partial_kernel;
using kernel::argmax_reduce_kernel;
using kernel::embedding_kernel;
using kernel::find_ngram_global_kernel;
using kernel::find_ngram_partial_kernel;
using kernel::linear_kernel;
using kernel::multitoken_paged_attention_task_impl;
using kernel::norm_linear_task_impl;
using kernel::paged_attention_task_impl;
using kernel::silu_mul_linear_task_impl;
using kernel::single_batch_decoding_kernel;
using kernel::single_batch_extend_kernel;
using kernel::single_batch_gqa_kernel;
using kernel::target_verify_greedy_kernel;
using bfloat16 = type::bfloat16_t;

// TODO: this file is too big to build fastly, so we may divide it into serveral files later.


// Linear

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
__global__ void linear_kernel_wrapper(void const *input_ptr,
                                      void const *weight_ptr,
                                      void const *residual_ptr,
                                      void *output_ptr) {
  linear_kernel<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, 4096>(
      input_ptr, weight_ptr, residual_ptr, output_ptr);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_linear(void const *input_ptr,
                   void const *weight_ptr,
                   void const *residual_ptr,
                   void *output_ptr) {
  dim3 grid_dim(85, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = 96 * 1024;

  constexpr int output_size = 48;
  cudaFuncSetAttribute(
      linear_kernel_wrapper<T, BATCH_SIZE, output_size, REDUCTION_SIZE>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);

  linear_kernel_wrapper<T, BATCH_SIZE, output_size, REDUCTION_SIZE>
      <<<grid_dim, block_dim, smem_size>>>(
          input_ptr, weight_ptr, residual_ptr, output_ptr);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
__global__ void tail_linear_kernel_wrapper(void const *input_ptr,
                                      void const *weight_ptr,
                                      void const *residual_ptr,
                                      void *output_ptr) {
  linear_kernel<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, 4096>(
      input_ptr, weight_ptr, residual_ptr, output_ptr);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_tail_linear(void const *input_ptr,
                   void const *weight_ptr,
                   void const *residual_ptr,
                   void *output_ptr) {
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = 96 * 1024;

  bfloat16 const * d_weight_ptr = static_cast<bfloat16 const *>(weight_ptr);
  bfloat16 * d_output_ptr = static_cast<bfloat16 *>(output_ptr);
  bfloat16 const * d_residual_ptr = static_cast<bfloat16 const *>(residual_ptr);
  d_weight_ptr += 48 * 4096 * 85;
  d_output_ptr += 48 * 1 * 85;
  d_residual_ptr += 48 * 1 * 85;

  weight_ptr = static_cast<void const *>(d_weight_ptr);
  output_ptr = static_cast<void *>(d_output_ptr);
  residual_ptr = static_cast<void const *>(d_residual_ptr);

  constexpr int output_size = 16;
  cudaFuncSetAttribute(
      tail_linear_kernel_wrapper<T, BATCH_SIZE, output_size, REDUCTION_SIZE>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);

  tail_linear_kernel_wrapper<T, BATCH_SIZE, output_size, REDUCTION_SIZE>
      <<<grid_dim, block_dim, smem_size>>>(
          input_ptr, weight_ptr, residual_ptr, output_ptr);
}

#define LINEAR_DISPATCH_OUTPUT_SIZE(BATCH_SIZE, OUTPUT_SIZE)                   \
  case OUTPUT_SIZE:                                                            \
    launch_linear<bfloat16, BATCH_SIZE, OUTPUT_SIZE, 4096>(                    \
        input_ptr, weight_ptr, residual_ptr, output_ptr);                      \
    break;

#define LINEAR_DISPATCH_BATCH_SIZE(BATCH_SIZE)                                 \
  case BATCH_SIZE:                                                             \
    switch (output.size(1)) {                                                  \
      LINEAR_DISPATCH_OUTPUT_SIZE(BATCH_SIZE, 16)                              \
      LINEAR_DISPATCH_OUTPUT_SIZE(BATCH_SIZE, 32)                              \
      LINEAR_DISPATCH_OUTPUT_SIZE(BATCH_SIZE, 64)                              \
      LINEAR_DISPATCH_OUTPUT_SIZE(BATCH_SIZE, 128)                              \
      LINEAR_DISPATCH_OUTPUT_SIZE(BATCH_SIZE, 1024)                              \
      LINEAR_DISPATCH_OUTPUT_SIZE(BATCH_SIZE, 4096)                              \
      LINEAR_DISPATCH_OUTPUT_SIZE(BATCH_SIZE, 48)                              \
      LINEAR_DISPATCH_OUTPUT_SIZE(BATCH_SIZE, 96)                              \
      LINEAR_DISPATCH_OUTPUT_SIZE(BATCH_SIZE, 112)                              \
      default:                                                                 \
        printf("Unsupported output size in test: %zu\n", output.size(1));      \
        break;                                                                 \
    }                                                                          \
    break;

void linear(torch::Tensor input,
            torch::Tensor weight,
            torch::Tensor residual,
            torch::Tensor output) {

  void const *input_ptr = input.data_ptr();
  void const *weight_ptr = weight.data_ptr();
  void const *residual_ptr = residual.data_ptr();
  void *output_ptr = output.data_ptr();

  switch (input.size(0)) {
    LINEAR_DISPATCH_BATCH_SIZE(1)
    LINEAR_DISPATCH_BATCH_SIZE(2)
    LINEAR_DISPATCH_BATCH_SIZE(3)
    LINEAR_DISPATCH_BATCH_SIZE(4)
    LINEAR_DISPATCH_BATCH_SIZE(5)
    LINEAR_DISPATCH_BATCH_SIZE(6)
    default:
      printf("Unsupported batch size in test: %zu\n", input.size(0));
      break;
  }

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

void tail_linear(torch::Tensor input,
            torch::Tensor weight,
            torch::Tensor residual,
            torch::Tensor output) {

  void const *input_ptr = input.data_ptr();
  void const *weight_ptr = weight.data_ptr();
  void const *residual_ptr = residual.data_ptr();
  void *output_ptr = output.data_ptr();

  launch_tail_linear<bfloat16, 1, 4096, 4096>(input_ptr, weight_ptr, residual_ptr, output_ptr);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}


// pybind11 bindings

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  // m.def("prompt_lookup", &prompt_lookup, "Prompt lookup kernel");
  // m.def("embedding", &embedding, "Embedding kernel");
  m.def("linear", &linear, "Linear kernel");
  m.def("tail_linear", &tail_linear, "Tail Linear kernel");
  // m.def("argmax", &argmax, "Argmax kernel");
  // m.def("verify", &verify, "Target verification kernel");
  // m.def("norm_linear", &norm_linear, "RMSNorm Linear kernel");
  // m.def("silu_mul_linear", &silu_mul_linear, "SILU MUL Linear kernel");
  // m.def("single_batch_decoding",
  //       &single_batch_decoding,
  //       py::arg("qkv"),
  //       py::arg("k_cache"),
  //       py::arg("v_cache"),
  //       py::arg("output"),
  //       py::arg("seq_len"),
  //       py::arg("qk_norm"),
  //       py::arg("rotary_embed"),
  //       py::arg("qnorm_weight") = py::none(),
  //       py::arg("knorm_weight") = py::none(),
  //       py::arg("cos") = py::none(),
  //       py::arg("sin") = py::none(),
  //       py::arg("q_eps") = 0.0f,
  //       py::arg("k_eps") = 0.0f);
  // m.def("single_batch_extend",
  //       &single_batch_extend,
  //       py::arg("qkv"),
  //       py::arg("k_cache"),
  //       py::arg("v_cache"),
  //       py::arg("output"),
  //       py::arg("seq_len"),
  //       py::arg("extend_num"),
  //       py::arg("qk_norm"),
  //       py::arg("rotary_embed"),
  //       py::arg("qnorm_weight") = py::none(),
  //       py::arg("knorm_weight") = py::none(),
  //       py::arg("cos") = py::none(),
  //       py::arg("sin") = py::none(),
  //       py::arg("q_eps") = 0.0f,
  //       py::arg("k_eps") = 0.0f,
  //       py::arg("q_norm_debug") = py::none(),
  //       py::arg("k_norm_debug") = py::none());
  // m.def("paged_attention", &paged_attention, "Paged Attention");
  // m.def("multitoken_paged_attention",
  //       &multitoken_paged_attention,
  //       "Multitoken Paged Attention");
  // m.def("rms_norm", &rms_norm, "Window RMSNorm");
}