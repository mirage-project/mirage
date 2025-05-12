#include "norm_linear.cuh"
#include <cuda_runtime.h>
#include <torch/extension.h>

using kernel::norm_linear_kernel;

template <typename T>
__global__ void norm_linear_kernel_wrapper(void const *input_ptr,
                                           void const *weight_ptr,
                                           void *output_ptr,
                                           int batch_size,
                                           int hidden_size) {
  norm_linear_kernel<T>(input_ptr, weight_ptr, output_ptr);
}

void launch_norm_linear(torch::Tensor input,
                        torch::Tensor weight,
                        torch::Tensor output) {

  void const *input_ptr = input.data_ptr();
  void const *weight_ptr = weight.data_ptr();
  void *output_ptr = output.data_ptr();

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  cudaFuncSetAttribute(norm_linear_kernel_wrapper<__nv_bfloat16>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       36666);
  norm_linear_kernel_wrapper<__nv_bfloat16><<<grid_dim, block_dim, 36666>>>(
      input_ptr, weight_ptr, output_ptr, input.size(0), input.size(1));
  cudaDeviceSynchronize();
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

// pybind11 bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("norm_linear", &launch_norm_linear, "RMSNorm Linear kernel");
}