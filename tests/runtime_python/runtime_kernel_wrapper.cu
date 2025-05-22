#include "norm_linear.cuh"
#include "silu_mul_linear.cuh"
#include <cuda_runtime.h>
#include <torch/extension.h>

using kernel::norm_linear_kernel;
using kernel::silu_mul_linear_kernel;
using bfloat16 = type::bfloat16_t;

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int SEQUENCE_SIZE>
__global__ void norm_linear_kernel_wrapper(void const *input_ptr,
                                           void const *weight_ptr,
                                           void *output_ptr) {
  norm_linear_kernel<T, BATCH_SIZE, OUTPUT_SIZE, SEQUENCE_SIZE>(
      input_ptr, weight_ptr, output_ptr);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int SEQUENCE_SIZE>
__global__ void silu_mul_linear_kernel_wrapper(void const *input_ptr,
                                               void const *mul_ptr,
                                               void const *weight_ptr,
                                               void *output_ptr) {
  silu_mul_linear_kernel<T, BATCH_SIZE, OUTPUT_SIZE, SEQUENCE_SIZE>(
      input_ptr, mul_ptr, weight_ptr, output_ptr);
}

void launch_norm_linear(torch::Tensor input,
                        torch::Tensor weight,
                        torch::Tensor output) {

  void const *input_ptr = input.data_ptr();
  void const *weight_ptr = weight.data_ptr();
  void *output_ptr = output.data_ptr();

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  int output_size = output.size(1);
  switch (output_size) {
    case 16:
      cudaFuncSetAttribute(norm_linear_kernel_wrapper<bfloat16, 1, 16, 3584>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           36666);
      norm_linear_kernel_wrapper<bfloat16, 1, 16, 3584>
          <<<grid_dim, block_dim, 36666>>>(input_ptr, weight_ptr, output_ptr);
      break;
    case 32:
      cudaFuncSetAttribute(norm_linear_kernel_wrapper<bfloat16, 1, 32, 3584>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           36666);
      norm_linear_kernel_wrapper<bfloat16, 1, 32, 3584>
          <<<grid_dim, block_dim, 36666>>>(input_ptr, weight_ptr, output_ptr);
      break;
    case 64:
      cudaFuncSetAttribute(norm_linear_kernel_wrapper<bfloat16, 1, 64, 3584>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           36666);
      norm_linear_kernel_wrapper<bfloat16, 1, 64, 3584>
          <<<grid_dim, block_dim, 36666>>>(input_ptr, weight_ptr, output_ptr);
      break;
    default:
      printf("Unsupported output size: %d\n", output_size);
      return;
  }
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
  int output_size = output.size(1);
  switch (output_size) {
    case 16:
      cudaFuncSetAttribute(
          silu_mul_linear_kernel_wrapper<bfloat16, 1, 16, 3584>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          36666);
      silu_mul_linear_kernel_wrapper<bfloat16, 1, 16, 3584>
          <<<grid_dim, block_dim, 36666>>>(
              input_ptr, mul_ptr, weight_ptr, output_ptr);
      break;
    case 32:
      cudaFuncSetAttribute(
          silu_mul_linear_kernel_wrapper<bfloat16, 1, 32, 3584>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          36666);
      silu_mul_linear_kernel_wrapper<bfloat16, 1, 32, 3584>
          <<<grid_dim, block_dim, 36666>>>(
              input_ptr, mul_ptr, weight_ptr, output_ptr);
      break;
    case 64:
      cudaFuncSetAttribute(
          silu_mul_linear_kernel_wrapper<bfloat16, 1, 64, 3584>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          36666);
      silu_mul_linear_kernel_wrapper<bfloat16, 1, 64, 3584>
          <<<grid_dim, block_dim, 36666>>>(
              input_ptr, mul_ptr, weight_ptr, output_ptr);
      break;
    default:
      printf("Unsupported output size: %d\n", output_size);
      return;
  }
  cudaDeviceSynchronize();
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

// pybind11 bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("norm_linear", &launch_norm_linear, "RMSNorm Linear kernel");
  m.def("silu_mul_linear", &silu_mul_linear, "SILU MUL Linear kernel");
}