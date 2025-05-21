#include "norm_linear.cuh"
#include "silu_mul_linear.cuh"
#include "single_decoding.cuh"
#include <cuda_runtime.h>
#include <torch/extension.h>

using kernel::norm_linear_kernel;
using kernel::silu_mul_linear_kernel;
using kernel::single_batch_decoding_kernel;
using bfloat16 = type::bfloat16_t;

template <typename T>
__global__ void single_batch_decoding_kernel_wrapper(void const *qkv_ptr,
                                                     void *k_cache_ptr,
                                                     void *v_cache_ptr,
                                                     void *output_ptr,
                                                     size_t seq_len) {
  single_batch_decoding_kernel<T, 64>(
      qkv_ptr, k_cache_ptr, v_cache_ptr, output_ptr);
}

template <typename T>
__global__ void norm_linear_kernel_wrapper(void const *input_ptr,
                                           void const *weight_ptr,
                                           void *output_ptr) {
  norm_linear_kernel<T>(input_ptr, weight_ptr, output_ptr);
}

template <typename T>
__global__ void silu_mul_linear_kernel_wrapper(void const *input_ptr,
                                               void const *mul_ptr,
                                               void const *weight_ptr,
                                               void *output_ptr) {
  silu_mul_linear_kernel<T>(input_ptr, mul_ptr, weight_ptr, output_ptr);
}

void norm_linear(torch::Tensor input,
                 torch::Tensor weight,
                 torch::Tensor output) {

  void const *input_ptr = input.data_ptr();
  void const *weight_ptr = weight.data_ptr();
  void *output_ptr = output.data_ptr();

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  cudaFuncSetAttribute(norm_linear_kernel_wrapper<bfloat16>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       36666);
  norm_linear_kernel_wrapper<bfloat16>
      <<<grid_dim, block_dim, 36666>>>(input_ptr, weight_ptr, output_ptr);
  cudaDeviceSynchronize();
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

void silu_mul_linear(torch::Tensor input,
                     torch::Tensor mul,
                     torch::Tensor weight,
                     torch::Tensor output) {

  void const *input_ptr = input.data_ptr();
  void const *mul_ptr = mul.data_ptr();
  void const *weight_ptr = weight.data_ptr();
  void *output_ptr = output.data_ptr();

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  cudaFuncSetAttribute(silu_mul_linear_kernel_wrapper<bfloat16>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       36666);
  silu_mul_linear_kernel_wrapper<bfloat16><<<grid_dim, block_dim, 36666>>>(
      input_ptr, mul_ptr, weight_ptr, output_ptr);
  cudaDeviceSynchronize();
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

void single_batch_decoding(torch::Tensor qkv,
                           torch::Tensor k_cache,
                           torch::Tensor v_cache,
                           torch::Tensor output,
                           size_t seq_len) {
  void const *qkv_ptr = qkv.data_ptr();
  void *k_cache_ptr = k_cache.data_ptr();
  void *v_cache_ptr = k_cache.data_ptr();
  void *output_ptr = output.data_ptr();
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  cudaFuncSetAttribute(single_batch_decoding_kernel_wrapper<bfloat16>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       36666);
  single_batch_decoding_kernel_wrapper<bfloat16>
      <<<grid_dim, block_dim, 36666>>>(
          qkv_ptr, k_cache_ptr, v_cache_ptr, output_ptr, seq_len);
  cudaDeviceSynchronize();
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

// pybind11 bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("norm_linear", &norm_linear, "RMSNorm Linear kernel");
  m.def("silu_mul_linear", &silu_mul_linear, "SILU MUL Linear kernel");
  m.def("single_batch_decoding",
        &single_batch_decoding,
        "FlashAttention Decoding kernel");
}