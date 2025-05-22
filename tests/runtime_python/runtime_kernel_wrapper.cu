#include "norm_linear.cuh"
#include "silu_mul_linear.cuh"
#include "single_batch_decoding.cuh"
#include <cuda_runtime.h>
#include <torch/extension.h>

using kernel::norm_linear_kernel;
using kernel::silu_mul_linear_kernel;
using kernel::single_batch_decoding_kernel;
using bfloat16 = type::bfloat16_t;

#define DISPATCH_SEQ_LEN(SEQ_LEN, FUNC, T, ...)                                \
  if ((SEQ_LEN) <= 8) {                                                        \
    FUNC<T, 8>(__VA_ARGS__);                                                   \
  } else if ((SEQ_LEN) <= 16) {                                                \
    FUNC<T, 16>(__VA_ARGS__);                                                  \
  } else if ((SEQ_LEN) <= 24) {                                                \
    FUNC<T, 24>(__VA_ARGS__);                                                  \
  } else if ((SEQ_LEN) <= 32) {                                                \
    FUNC<T, 32>(__VA_ARGS__);                                                  \
  } else if ((SEQ_LEN) <= 40) {                                                \
    FUNC<T, 40>(__VA_ARGS__);                                                  \
  } else if ((SEQ_LEN) <= 48) {                                                \
    FUNC<T, 48>(__VA_ARGS__);                                                  \
  } else if ((SEQ_LEN) <= 56) {                                                \
    FUNC<T, 56>(__VA_ARGS__);                                                  \
  } else if ((SEQ_LEN) <= 64) {                                                \
    FUNC<T, 64>(__VA_ARGS__);                                                  \
  } else if ((SEQ_LEN) <= 72) {                                                \
    FUNC<T, 72>(__VA_ARGS__);                                                  \
  } else if ((SEQ_LEN) <= 80) {                                                \
    FUNC<T, 80>(__VA_ARGS__);                                                  \
  } else if ((SEQ_LEN) <= 88) {                                                \
    FUNC<T, 88>(__VA_ARGS__);                                                  \
  } else if ((SEQ_LEN) <= 96) {                                                \
    FUNC<T, 96>(__VA_ARGS__);                                                  \
  } else if ((SEQ_LEN) <= 104) {                                               \
    FUNC<T, 104>(__VA_ARGS__);                                                 \
  } else if ((SEQ_LEN) <= 112) {                                               \
    FUNC<T, 112>(__VA_ARGS__);                                                 \
  } else if ((SEQ_LEN) <= 120) {                                               \
    FUNC<T, 120>(__VA_ARGS__);                                                 \
  } else if ((SEQ_LEN) <= 128) {                                               \
    FUNC<T, 128>(__VA_ARGS__);                                                 \
  } else {                                                                     \
    printf("Unsupported seq_len: %zu\n", SEQ_LEN);                             \
  }

template <typename T, size_t SEQ_LEN>
__global__ void single_batch_decoding_kernel_wrapper(void const *qkv_ptr,
                                                     void *k_cache_ptr,
                                                     void *v_cache_ptr,
                                                     void *output_ptr) {
  single_batch_decoding_kernel<T, SEQ_LEN>(
      qkv_ptr, k_cache_ptr, v_cache_ptr, output_ptr);
}

template <typename T, size_t SEQ_LEN>
void launch_single_batch_decoding(void const *qkv_ptr,
                                  void *k_cache_ptr,
                                  void *v_cache_ptr,
                                  void *output_ptr) {
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = 76672;

  cudaFuncSetAttribute(single_batch_decoding_kernel_wrapper<T, SEQ_LEN>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_size);

  single_batch_decoding_kernel_wrapper<T, SEQ_LEN>
      <<<grid_dim, block_dim, smem_size>>>(
          qkv_ptr, k_cache_ptr, v_cache_ptr, output_ptr);
}

void single_batch_decoding(torch::Tensor qkv,
                           torch::Tensor k_cache,
                           torch::Tensor v_cache,
                           torch::Tensor output,
                           size_t seq_len) {

  void const *qkv_ptr = qkv.data_ptr();
  void *k_cache_ptr = k_cache.data_ptr();
  void *v_cache_ptr = v_cache.data_ptr();
  void *output_ptr = output.data_ptr();

  DISPATCH_SEQ_LEN(seq_len,
                   launch_single_batch_decoding,
                   bfloat16,
                   qkv_ptr,
                   k_cache_ptr,
                   v_cache_ptr,
                   output_ptr);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

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

void norm_linear(torch::Tensor input,
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
  m.def("norm_linear", &norm_linear, "RMSNorm Linear kernel");
  m.def("silu_mul_linear", &silu_mul_linear, "SILU MUL Linear kernel");
  m.def("single_batch_decoding",
        &single_batch_decoding,
        "FlashAttention Decoding kernel");
}