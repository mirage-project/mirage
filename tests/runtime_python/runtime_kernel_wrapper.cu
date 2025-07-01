#include "argmax.cuh"
#include "linear.cuh"
#include "norm_linear.cuh"
#include "silu_mul_linear.cuh"
#include "single_batch_decoding.cuh"
#include "single_batch_gqa.cuh"
#include "embedding.cuh"
#include "prompt_lookup.cuh"
#include "bfloat16.h"
#include <cuda_runtime.h>
#include <torch/extension.h>

// using kernel::argmax_kernel;
using kernel::linear_kernel;
using kernel::norm_linear_task_impl;
using kernel::silu_mul_linear_task_impl;
using kernel::single_batch_decoding_kernel;
using kernel::single_batch_gqa_kernel;
using kernel::embedding_kernel;
using kernel::find_ngram_partial_kernel;
using kernel::find_ngram_global_kernel;
using bfloat16 = type::bfloat16_t;

template <typename T>
__global__ void single_batch_gqa_kernel_wrapper(void const *qkv_ptr,
                                                void *k_cache_ptr,
                                                void *v_cache_ptr,
                                                void *output_ptr,
                                                size_t seq_len,
                                                bool qk_norm,
                                                bool rotary_embed,
                                                void const *qnorm_weight_ptr,
                                                void const *knorm_weight_ptr,
                                                void const *cos_ptr,
                                                void const *sin_ptr,
                                                float q_eps,
                                                float k_eps) {
  single_batch_gqa_kernel<T, 4>(qkv_ptr,
                                k_cache_ptr,
                                v_cache_ptr,
                                output_ptr,
                                seq_len,
                                qk_norm,
                                rotary_embed,
                                qnorm_weight_ptr,
                                knorm_weight_ptr,
                                cos_ptr,
                                sin_ptr,
                                q_eps,
                                k_eps);
}

void single_batch_gqa(
    torch::Tensor qkv,
    torch::Tensor k_cache,
    torch::Tensor v_cache,
    torch::Tensor output,
    size_t seq_len,
    bool qk_norm,
    bool rotary_embed,
    torch::optional<torch::Tensor> qnorm_weight = torch::nullopt,
    torch::optional<torch::Tensor> knorm_weight = torch::nullopt,
    torch::optional<torch::Tensor> cos = torch::nullopt,
    torch::optional<torch::Tensor> sin = torch::nullopt,
    float q_eps = 0.0f,
    float k_eps = 0.0f) {
  void const *qkv_ptr = qkv.data_ptr();
  void *k_cache_ptr = k_cache.data_ptr();
  void *v_cache_ptr = v_cache.data_ptr();
  void *output_ptr = output.data_ptr();

  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = 88888;

  void const *qnorm_weight_ptr = qk_norm ? qnorm_weight->data_ptr() : nullptr;
  void const *knorm_weight_ptr = qk_norm ? knorm_weight->data_ptr() : nullptr;
  void const *cos_ptr = rotary_embed ? cos->data_ptr() : nullptr;
  void const *sin_ptr = rotary_embed ? sin->data_ptr() : nullptr;

  cudaFuncSetAttribute(single_batch_gqa_kernel_wrapper<bfloat16>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_size);

  single_batch_gqa_kernel_wrapper<bfloat16>
      <<<grid_dim, block_dim, smem_size>>>(qkv_ptr,
                                           k_cache_ptr,
                                           v_cache_ptr,
                                           output_ptr,
                                           seq_len,
                                           qk_norm,
                                           rotary_embed,
                                           qnorm_weight_ptr,
                                           knorm_weight_ptr,
                                           cos_ptr,
                                           sin_ptr,
                                           q_eps,
                                           k_eps);

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

// RMSNorm Linear

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
__global__ void norm_linear_kernel_wrapper(void const *input_ptr,
                                           void const *norm_weight_ptr,
                                           void const *weight_ptr,
                                           float eps,
                                           void *output_ptr) {
  norm_linear_task_impl<T,
                        BATCH_SIZE,
                        OUTPUT_SIZE,
                        REDUCTION_SIZE,
                        OUTPUT_SIZE>(
      input_ptr, norm_weight_ptr, weight_ptr, eps, output_ptr);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_norm_linear(void const *input_ptr,
                        void const *norm_weight_ptr,
                        void const *weight_ptr,
                        float eps,
                        void *output_ptr) {
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = 112640;

  cudaFuncSetAttribute(
      norm_linear_kernel_wrapper<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);

  norm_linear_kernel_wrapper<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>
      <<<grid_dim, block_dim, smem_size>>>(
          input_ptr, norm_weight_ptr, weight_ptr, eps, output_ptr);
}

void norm_linear(torch::Tensor input,
                 torch::Tensor norm_weight,
                 torch::Tensor weight,
                 float eps,
                 torch::Tensor output) {

  void const *input_ptr = input.data_ptr();
  void const *norm_weight_ptr = norm_weight.data_ptr();
  void const *weight_ptr = weight.data_ptr();
  void *output_ptr = output.data_ptr();

  switch (output.size(1)) {
    case 16:
      launch_norm_linear<bfloat16, 1, 16, 4096>(input_ptr, norm_weight_ptr,
                                                weight_ptr, eps, output_ptr);
      break;
    case 32:
      launch_norm_linear<bfloat16, 1, 32, 4096>(input_ptr, norm_weight_ptr,
                                                weight_ptr, eps, output_ptr);
      break;
    case 64:
      launch_norm_linear<bfloat16, 1, 64, 4096>(input_ptr, norm_weight_ptr,
                                                weight_ptr, eps, output_ptr);
      break;
    case 256:
      launch_norm_linear<bfloat16, 1, 256, 4096>(input_ptr, norm_weight_ptr,
                                                weight_ptr, eps, output_ptr);
      break;
    case 1600:
      launch_norm_linear<bfloat16, 1, 1600, 4096>(input_ptr, norm_weight_ptr,
                                                weight_ptr, eps, output_ptr);
      break;
    default:
      printf("Unsupported output size in test: %zu\n", output.size(1));
      break;
  }

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

// SiLU MUL Linear

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
__global__ void silu_mul_linear_kernel_wrapper(void const *input_ptr,
                                               void const *weight_ptr,
                                               void const *bias_ptr,
                                               void *output_ptr) {
  silu_mul_linear_task_impl<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>(
      input_ptr, weight_ptr, bias_ptr, output_ptr);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_silu_mul_linear(void const *input_ptr,
                            void const *weight_ptr,
                            void const *bias_ptr,
                            void *output_ptr) {
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = 112640;

  cudaFuncSetAttribute(silu_mul_linear_kernel_wrapper<T,
                                                      BATCH_SIZE,
                                                      OUTPUT_SIZE,
                                                      REDUCTION_SIZE>,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_size);

  silu_mul_linear_kernel_wrapper<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>
      <<<grid_dim, block_dim, smem_size>>>(
          input_ptr, weight_ptr, bias_ptr, output_ptr);
}

void silu_mul_linear(torch::Tensor input,
                     torch::Tensor weight,
                     torch::Tensor bias,
                     torch::Tensor output) {

  void const *input_ptr = input.data_ptr();
  void const *weight_ptr = weight.data_ptr();
  void const *bias_ptr = bias.data_ptr();
  void *output_ptr = output.data_ptr();

  switch (output.size(1)) {
    case 16:
      launch_silu_mul_linear<bfloat16, 1, 16, 12288>(input_ptr, weight_ptr,
                                                     bias_ptr, output_ptr);
      break;
    case 32:
      launch_silu_mul_linear<bfloat16, 1, 32, 12288>(input_ptr, weight_ptr,
                                                     bias_ptr, output_ptr);
      break;
    case 64:
      launch_silu_mul_linear<bfloat16, 1, 64, 12288>(input_ptr, weight_ptr,
                                                     bias_ptr, output_ptr);
      break;
    default:
      printf("Unsupported output size in test: %zu\n", output.size(1));
      break;
  }

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

// Linear

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
__global__ void linear_kernel_wrapper(void const *input_ptr,
                                      void const *weight_ptr,
                                      void const *residual_ptr,
                                      void *output_ptr) {
  linear_kernel<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>(
      input_ptr, weight_ptr, residual_ptr, output_ptr);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_linear(void const *input_ptr,
                   void const *weight_ptr,
                   void const *residual_ptr,
                   void *output_ptr) {
  dim3 grid_dim(1, 1, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = 112640;

  cudaFuncSetAttribute(
      linear_kernel_wrapper<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);

  linear_kernel_wrapper<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE>
      <<<grid_dim, block_dim, smem_size>>>(
          input_ptr, weight_ptr, residual_ptr, output_ptr);
}

void linear(torch::Tensor input,
            torch::Tensor weight,
            torch::Tensor residual,
            torch::Tensor output) {

  void const *input_ptr = input.data_ptr();
  void const *weight_ptr = weight.data_ptr();
  void const *residual_ptr = residual.data_ptr();
  void *output_ptr = output.data_ptr();

  switch (output.size(1)) {
    case 16:
      launch_linear<bfloat16, 1, 16, 4096>(input_ptr, weight_ptr,
                                           residual_ptr, output_ptr);
      break;
    case 32:
      launch_linear<bfloat16, 1, 32, 4096>(input_ptr, weight_ptr,
                                           residual_ptr, output_ptr);
      break;
    case 64:
      launch_linear<bfloat16, 1, 64, 4096>(input_ptr, weight_ptr,
                                           residual_ptr, output_ptr);
      break;
    default:
      printf("Unsupported output size in test: %zu\n", output.size(1));
      break;
  }

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
  }
}

// Argmax

// template <typename T>
// __global__ void argmax_kernel_wrapper(void const *input_ptr, void
// *output_ptr) {
//   argmax_kernel<T, 1, 32768>(input_ptr, output_ptr);
// }

// template <typename T>
// void launch_argmax(void const *input_ptr, void *output_ptr) {
//   dim3 grid_dim(1, 1, 1);
//   dim3 block_dim(128, 1, 1);
//   size_t smem_size = 36666;

//   cudaFuncSetAttribute(argmax_kernel_wrapper<T>,
//                        cudaFuncAttributeMaxDynamicSharedMemorySize,
//                        smem_size);

//   argmax_kernel_wrapper<T>
//       <<<grid_dim, block_dim, smem_size>>>(input_ptr, output_ptr);
// }

// void argmax(torch::Tensor input, torch::Tensor output) {

//   void const *input_ptr = input.data_ptr();
//   void *output_ptr = output.data_ptr();

//   launch_argmax<bfloat16>(input_ptr, output_ptr);

//   cudaError_t err = cudaDeviceSynchronize();
//   if (err != cudaSuccess) {
//     printf("CUDA kernel launch error: %s\n", cudaGetErrorString(err));
//   }
// }

// Embedding Kernel
template <typename T, int CHUNK_SIZE, int OUTPUT_DIM_SIZE>
__global__ void embedding_kernel_wrapper(void const *input_ptr,
                                         void const *embedding_ptr,
                                         void *output_ptr) {
  int input_offset = blockIdx.x;
  int64_t const *__restrict__ input = static_cast<int64_t const *>(input_ptr) + input_offset;
  int embedding_offset = blockIdx.y * CHUNK_SIZE;
  T const *__restrict__ embedding = static_cast<T const *>(embedding_ptr) + embedding_offset;
  int output_offset = blockIdx.y * CHUNK_SIZE + blockIdx.x * OUTPUT_DIM_SIZE;
  T *__restrict__ output = static_cast<T *>(output_ptr) + output_offset;

  if (blockIdx.x == 1 && blockIdx.y == 1 && threadIdx.x == 0) {
    printf("input_offset: %d, embedding_offset: %d, output_offset: %d\n", input_offset, embedding_offset, output_offset);
  }
  embedding_kernel<T, CHUNK_SIZE, OUTPUT_DIM_SIZE>(
      input, embedding, output);
  // if (blockIdx.x == 1 && blockIdx.y == 1) {
  //   printf("input: %d, embedding: %d, output: %d\n", input, embedding, output);
  // }
}

void embedding(torch::Tensor input,
               torch::Tensor weight,
               torch::Tensor output) {

  dim3 grid_dim(input.size(1), output.size(1) / 128, 1);
  printf("grid_dim: %d, %d, %d\n", grid_dim.x, grid_dim.y, grid_dim.z);
  dim3 block_dim(128, 1, 1);

  embedding_kernel_wrapper<float, 128, 4096><<<grid_dim, block_dim>>>(
      input.data_ptr(),
      weight.data_ptr(),
      output.data_ptr());

  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error in embedding: %s\n",
           cudaGetErrorString(err));
  }
}

// Prompt Lookup Kernel
template <int NGRAM_SIZE, int NUM_WORKERS>
__global__ void find_ngram_partial_kernel_wrapper(long long const *__restrict__ input_ptr,
                                                  long long *__restrict__ output_id_ptr,
                                                  int input_token_num) {
  // Each block gets a pointer to its unique output slot.
  long long *block_output_ptr = output_id_ptr + blockIdx.x;
  find_ngram_partial_kernel<NGRAM_SIZE, NUM_WORKERS>(input_ptr, block_output_ptr, input_token_num);
}

template <int NGRAM_SIZE, int SPEC_LENGTH, int NUM_PARTIAL_TASKS>
__global__ void find_ngram_global_kernel_wrapper(long long const *__restrict__ input_array,
                                                 long long const *__restrict__ tokens_ptr,
                                                 long long *__restrict__ output_result,
                                                 int step) {
  find_ngram_global_kernel<NGRAM_SIZE, SPEC_LENGTH, NUM_PARTIAL_TASKS>(input_array, tokens_ptr, output_result, step);
}

void prompt_lookup(torch::Tensor all_tokens,
                   int prompt_len,
                   int ngram_size,
                   int spec_length,
                   torch::Tensor final_output) {
  
  constexpr int NUM_WORKERS = 96; // Corresponds to grid size
  dim3 partial_grid_dim(NUM_WORKERS, 1, 1);
  dim3 partial_block_dim(128, 1, 1);
  
  auto partial_output_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA);
  torch::Tensor partial_output = torch::full({NUM_WORKERS}, INT_MAX, partial_output_options);
  
  if (ngram_size == 3) {
    find_ngram_partial_kernel_wrapper<3, NUM_WORKERS><<<partial_grid_dim, partial_block_dim>>>(
        static_cast<long long const *>(all_tokens.data_ptr()),
        static_cast<long long *>(partial_output.data_ptr()),
        prompt_len);
  } else {
    throw std::runtime_error("Unsupported ngram_size for prompt_lookup test");
  }

  dim3 global_grid_dim(1, 1, 1);
  dim3 global_block_dim(128, 1, 1);

  if (ngram_size == 3 && spec_length == 5) {
     find_ngram_global_kernel_wrapper<3, 5, NUM_WORKERS><<<global_grid_dim, global_block_dim>>>(
        static_cast<long long const *>(partial_output.data_ptr()),
        static_cast<long long const *>(all_tokens.data_ptr()),
        static_cast<long long *>(final_output.data_ptr()),
        prompt_len);
  } else {
     throw std::runtime_error("Unsupported ngram_size/spec_length for prompt_lookup test");
  }
  
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    printf("CUDA kernel launch error in prompt_lookup: %s\n",
           cudaGetErrorString(err));
  }
}

// pybind11 bindings

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("prompt_lookup", &prompt_lookup, "Prompt lookup kernel");
  m.def("embedding", &embedding, "Embedding kernel");
  m.def("linear", &linear, "Linear kernel");
  // m.def("argmax", &argmax, "argmax kernel");
  m.def("norm_linear", &norm_linear, "RMSNorm Linear kernel");
  m.def("silu_mul_linear", &silu_mul_linear, "SILU MUL Linear kernel");
  // m.def("single_batch_gqa", &single_batch_gqa, "Decoding kernel");
  m.def("single_batch_gqa",
        &single_batch_gqa,
        py::arg("qkv"),
        py::arg("k_cache"),
        py::arg("v_cache"),
        py::arg("output"),
        py::arg("seq_len"),
        py::arg("qk_norm"),
        py::arg("rotary_embed"),
        py::arg("qnorm_weight") = py::none(),
        py::arg("knorm_weight") = py::none(),
        py::arg("cos") = py::none(),
        py::arg("sin") = py::none(),
        py::arg("q_eps") = 0.0f,
        py::arg("k_eps") = 0.0f);
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
  // m.def("single_batch_decoding",
  //       &single_batch_decoding,
  //       "FlashAttention Decoding kernel");
}