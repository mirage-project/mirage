#include <cuda_runtime.h>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <cmath>

#include "../include/mirage/config.h"
// #include "../include/mirage/persistent_kernel/tasks/linear_old.cuh"
// #include "../include/mirage/persistent_kernel/tasks/linear_reg.cuh"
// #include "../include/mirage/persistent_kernel/tasks/linear_reg_mem.cuh"
// #include "../include/mirage/persistent_kernel/tasks/linear_3d.cuh"
// #include "../include/mirage/persistent_kernel/tasks/linear_3d_ld.cuh"
// #include "../include/mirage/persistent_kernel/tasks/linear.cuh"
#include "../include/mirage/persistent_kernel/tasks/linear_3d_ld_seq.cuh"

// #define LINEAR_MPK
// #ifdef LINEAR_MPK
// #include "../include/mirage/persistent_kernel/tasks/linear_mpk.cuh"
// #endif

#define CUDA_CHECK(call)                                                 \
  do {                                                                   \
    cudaError_t err = call;                                              \
    if (err != cudaSuccess) {                                            \
      fprintf(stderr, "CUDA error at %s %d: %s\n", __FILE__, __LINE__,   \
              cudaGetErrorString(err));                                  \
      exit(EXIT_FAILURE);                                                \
    }                                                                    \
  } while (0)


// #define __CORRECTNESS_TEST__
#ifdef __CORRECTNESS_TEST__

// A simple, straightforward CPU implementation of Linear with residual addition.
// This function prioritizes correctness and clarity over performance.
template <typename T>
void cpu_linear_with_residual(
    const std::vector<T>& input,    // shape: [BATCH_SIZE, REDUCTION_SIZE]
    const std::vector<T>& weight,   // shape: [OUTPUT_SIZE, REDUCTION_SIZE]
    const std::vector<T>& residual, // shape: [BATCH_SIZE, OUTPUT_SIZE]
    std::vector<T>& output,         // shape: [BATCH_SIZE, OUTPUT_SIZE]
    int BATCH_SIZE,
    int OUTPUT_SIZE,
    int REDUCTION_SIZE)
{
    // The layout of weight is [OUTPUT_SIZE, REDUCTION_SIZE], which is already
    // what we need for a dot product (no transpose needed).
    for (int b = 0; b < BATCH_SIZE; ++b) {
        for (int o = 0; o < OUTPUT_SIZE; ++o) {
            // Pointer to the start of the current row in input matrix
            const T* input_row = &input[b * REDUCTION_SIZE];
            // Pointer to the start of the current row in weight matrix
            const T* weight_row = &weight[o * REDUCTION_SIZE];

            // Perform dot product using float for higher precision accumulation
            float accumulator = 0.0f;
            for (int k = 0; k < REDUCTION_SIZE; ++k) {
                accumulator += static_cast<float>(input_row[k]) * static_cast<float>(weight_row[k]);
            }

            // Add residual and cast back to the target type T
            float result = accumulator + static_cast<float>(residual[b * OUTPUT_SIZE + o]);
            output[b * OUTPUT_SIZE + o] = T(result);
        }
    }
}

// Function to compare two vectors and report differences
template <typename T>
bool compare_results(const std::vector<T>& ref_output, const std::vector<T>& kernel_output, int BATCH_SIZE, int OUTPUT_SIZE) {
    bool passed = true;
    double max_abs_err = 0.0;
    double max_rel_err = 0.0;

    for (int i = 0; i < BATCH_SIZE * OUTPUT_SIZE; ++i) {
        float ref_val = static_cast<float>(ref_output[i]);
        float kernel_val = static_cast<float>(kernel_output[i]);

        double abs_err = std::abs(ref_val - kernel_val);
        double rel_err = (ref_val == 0) ? abs_err : abs_err / std::abs(ref_val);

        max_abs_err = std::max(max_abs_err, abs_err);
        max_rel_err = std::max(max_rel_err, rel_err);
        
        // Define a tolerance for bfloat16 comparisons. 
        // A small absolute error is acceptable due to precision differences.
        const float tolerance = 1e-2; 
        if (abs_err > tolerance) {
            if (passed) { // Print header only once
                std::cout << "Correctness test FAILED!" << std::endl;
                std::cout << std::fixed << std::setprecision(5);
                std::cout << "First mismatch found at index " << i << " (row " << i / OUTPUT_SIZE << ", col " << i % OUTPUT_SIZE << "):" << std::endl;
            }
            std::cout << "  - CPU Reference: " << ref_val << std::endl;
            std::cout << "  - GPU Kernel:    " << kernel_val << std::endl;
            std::cout << "  - Absolute Error: " << abs_err << std::endl;
            std::cout << "  - Relative Error: " << rel_err << std::endl;
            passed = false;
            return passed; // Exit after first error for brevity
        }
    }

    if (passed) {
        std::cout << "Correctness test PASSED!" << std::endl;
        std::cout << std::fixed << std::setprecision(5);
        std::cout << "  - Max Absolute Error: " << max_abs_err << std::endl;
        std::cout << "  - Max Relative Error: " << max_rel_err << std::endl;
    }
    return passed;
}

#endif
// =================================================================

#ifdef LINEAR_MPK
template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int O_STRIDE,
          int K_PIPE_MAX>
__global__ void linear_kernel_launcher(void const *input_ptr,
                                       void const *weight_ptr,
                                       void const *residual_ptr,
                                       void *output_ptr,
                                       int num_active_tokens,
                                       bool use_residual) {
  kernel::linear_kernel<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, O_STRIDE, K_PIPE_MAX>(
      input_ptr, weight_ptr, residual_ptr, output_ptr, use_residual);
}
#else
template <typename T,
          int BATCH_SIZE,
          int OUTPUT_SIZE,
          int REDUCTION_SIZE,
          int O_STRIDE,
          int K_PIPE_MAX>
__global__ void linear_kernel_launcher(void const *input_ptr,
                                       void const *weight_ptr,
                                       void const *residual_ptr,
                                       void *output_ptr,
                                       int num_active_tokens,
                                       bool use_residual) {
  kernel::linear_kernel<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, O_STRIDE, K_PIPE_MAX>(
      input_ptr, weight_ptr, residual_ptr, output_ptr, num_active_tokens, use_residual);
}
#endif

int main() {
  using T = type::bfloat16_t;

  std::cout << "Starting test_linear" << std::endl;
  constexpr int BATCH_SIZE = 1;       // Must be <= 16 (NUM_ITERS_M == 1)
  constexpr int OUTPUT_SIZE = 64;     // Use 128 to match one atom in linear
  constexpr int REDUCTION_SIZE = 4096; // Must be multiple of 128
  constexpr int O_STRIDE = OUTPUT_SIZE;
  constexpr int K_PIPE_MAX = 3;

  const int num_active_tokens = BATCH_SIZE;

  // Host buffers
  std::vector<T> h_input(BATCH_SIZE * REDUCTION_SIZE);
  std::vector<T> h_weight(OUTPUT_SIZE * REDUCTION_SIZE);
  std::vector<T> h_residual(BATCH_SIZE * OUTPUT_SIZE);
  std::vector<T> h_output(BATCH_SIZE * OUTPUT_SIZE);

  // Initialize with deterministic pseudo-random data
  srand(42);
  auto fill_vec = [](std::vector<T> &v) {
    for (size_t i = 0; i < v.size(); ++i) {
      float r = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      v[i] = T(r);
    }
  };
  auto fill_vec_with = [](std::vector<T> &v, T value) {
    for (size_t i = 0; i < v.size(); ++i) {
      v[i] = value;
    }
  };
  // fill_vec(h_input);
  // fill_vec(h_weight);
  // fill_vec(h_residual);
  fill_vec_with(h_input, T(0));
  fill_vec_with(h_weight, T(0));
  fill_vec(h_residual);

  // Device buffers
  T *d_input = nullptr;
  T *d_weight = nullptr;
  T *d_residual = nullptr;
  T *d_output = nullptr;
  
  CUDA_CHECK(cudaMalloc(&d_input, sizeof(T) * h_input.size()));
  CUDA_CHECK(cudaMalloc(&d_weight, sizeof(T) * h_weight.size()));
  CUDA_CHECK(cudaMalloc(&d_residual, sizeof(T) * h_residual.size()));
  CUDA_CHECK(cudaMalloc(&d_output, sizeof(T) * h_output.size()));
  
  CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), sizeof(T) * h_input.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), sizeof(T) * h_weight.size(), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_residual, h_residual.data(), sizeof(T) * h_residual.size(), cudaMemcpyHostToDevice));

  
  std::cout << "Device memory allocated" << std::endl;

  // Kernel configuration
  dim3 gridDim(1);
  dim3 blockDim(128);
  size_t shared_mem_size = mirage::runtime::MAX_SHARE_MEMORY_SIZE;

  CUDA_CHECK(cudaFuncSetAttribute(linear_kernel_launcher<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, O_STRIDE, K_PIPE_MAX>,
     cudaFuncAttributeMaxDynamicSharedMemorySize, shared_mem_size));
  // Warmup
  linear_kernel_launcher<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, O_STRIDE, K_PIPE_MAX>
      <<<gridDim, blockDim, shared_mem_size>>>(
          d_input, d_weight, d_residual, d_output, num_active_tokens, /*use_residual=*/true);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, sizeof(T) * h_output.size(), cudaMemcpyDeviceToHost));

  // std::cout << "Output: " << std::endl;
  // for (size_t row = 0; row < BATCH_SIZE; ++row) {
  //   for (size_t col = 0; col < OUTPUT_SIZE; ++col) {
  //     std::cout << float(h_output[row * OUTPUT_SIZE + col]) << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << std::endl;

  // std::cout << "Residual: " << std::endl;
  // for (size_t row = 0; row < BATCH_SIZE; ++row) {
  //   for (size_t col = 0; col < OUTPUT_SIZE; ++col) {
  //     std::cout << float(h_residual[row * OUTPUT_SIZE + col]) << " ";
  //   }
  //   std::cout << std::endl;
  // }
  // std::cout << std::endl;

#ifdef __CORRECTNESS_TEST__
  std::cout << "\n--- Running Correctness Test ---" << std::endl;
  std::vector<T> h_output_ref(BATCH_SIZE * OUTPUT_SIZE);

  std::cout << "Calculating reference solution on CPU..." << std::endl;
  cpu_linear_with_residual(h_input, h_weight, h_residual, h_output_ref, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE);
  std::cout << "CPU calculation finished." << std::endl;

  std::cout << "Comparing CPU reference with GPU kernel output..." << std::endl;
  compare_results(h_output_ref, h_output, BATCH_SIZE, OUTPUT_SIZE);
  std::cout << "--- Correctness Test Finished ---\n" << std::endl;
#endif
  // Timing
//   int const num_runs = 100;
//   cudaEvent_t start, stop;
//   CUDA_CHECK(cudaEventCreate(&start));
//   CUDA_CHECK(cudaEventCreate(&stop));

//   CUDA_CHECK(cudaEventRecord(start));
//   for (int i = 0; i < num_runs; ++i) {
//     linear_kernel_launcher<T, BATCH_SIZE, OUTPUT_SIZE, REDUCTION_SIZE, O_STRIDE, K_PIPE_MAX>
//         <<<gridDim, blockDim, shared_mem_size>>>(
//             d_input, d_weight, d_residual, d_output, num_active_tokens, /*use_residual=*/true);
//   }
//   CUDA_CHECK(cudaEventRecord(stop));
//   CUDA_CHECK(cudaEventSynchronize(stop));

//   float ms = 0.0f;
//   CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
//   std::cout << "Average kernel time over " << num_runs << " runs: " << (ms / num_runs) << " ms\n";

//   // Copy back one run's output for a quick sanity print
//   CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, sizeof(T) * h_output.size(), cudaMemcpyDeviceToHost));
//   std::cout << "Output[0..7]: ";
//   for (int i = 0; i < 8 && i < (int)h_output.size(); ++i) {
//     std::cout << float(h_output[i]) << (i + 1 < 8 ? ", " : "\n");
//   }

  // Cleanup
//   CUDA_CHECK(cudaEventDestroy(start));
//   CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_weight));
  CUDA_CHECK(cudaFree(d_residual));
  CUDA_CHECK(cudaFree(d_output));


  return 0;
}


