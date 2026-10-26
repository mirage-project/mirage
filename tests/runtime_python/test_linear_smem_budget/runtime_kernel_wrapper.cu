#include "runtime_header.h"
#include "tasks/ampere/linear_cutlass.cuh"
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

using bfloat16 = type::bfloat16_t;

// One 128-wide output tile, as in the qwen3 lm_head task from #794.
constexpr int BATCH_SIZE = 1;
constexpr int OUTPUT_SIZE = 128;
constexpr int REDUCTION_SIZE = 1024;

__global__ void
    linear_wrapper(void const *input, void const *weight, void *output) {
  kernel::linear_kernel<bfloat16, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>(
      input, weight, nullptr, output, BATCH_SIZE, false);
}

void linear(torch::Tensor input, torch::Tensor weight, torch::Tensor output) {
  TORCH_CHECK(input.size(0) == BATCH_SIZE && input.size(1) == REDUCTION_SIZE);
  TORCH_CHECK(weight.size(0) == OUTPUT_SIZE);
  // Launch with exactly the dynamic shared memory a worker gets.
  size_t smem_size = mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE;
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      linear_wrapper, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
  linear_wrapper<<<1, 128, smem_size>>>(
      input.data_ptr(), weight.data_ptr(), output.data_ptr());
  C10_CUDA_CHECK(cudaDeviceSynchronize());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("linear", &linear);
}
