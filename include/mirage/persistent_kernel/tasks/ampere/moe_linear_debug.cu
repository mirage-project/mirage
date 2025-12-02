// Compile this file with:
// nvcc --expt-relaxed-constexpr -std=c++17 -O3 -arch=sm_80 -I../../../../../deps/cutlass/include moe_linear_debug.cu --ptxas-options=-v -o moe_cutlass.o
// or
// nvcc --expt-relaxed-constexpr -std=c++17 -O3 -arch=sm_80 -I/home/wenqin/study/cutlass/include moe_linear_debug.cu --ptxas-options=-v -o moe_cutlass.o

// Run it with:
// ./moe_cutlass.o --iters 1


#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <unordered_set>

constexpr int batch_size = 8;
constexpr int experts_size = 128;
constexpr int activate_experts_size = 8;

#define MIRAGE_UNIT_TEST 1

#include "./moe_linear.cuh"

// Matrix dimensions
const int m = batch_size;
const int k = 2048;
// const int n = 4096;
const int n = 1536;

const int expert_stride = 5;

using bfloat16 = __nv_bfloat16;


template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int OUTPUT_STRIDE, int REDUCTION_SIZE>
__global__ void moe_kernel_wrapper(void const *input_ptr,
                                      void const *weight_ptr,
                                      void const *residual_ptr,
                                      void *output_ptr,
                                      void const *expert_routing,
                                      void const *expert_mask) {
  kernel::moe_linear_kernel<T, BATCH_SIZE, OUTPUT_SIZE, OUTPUT_STRIDE, REDUCTION_SIZE, experts_size, activate_experts_size, expert_stride, true, true>(
      input_ptr, weight_ptr, residual_ptr, output_ptr, expert_routing, expert_mask, blockIdx.x);
}

template <typename T, int BATCH_SIZE, int OUTPUT_SIZE, int REDUCTION_SIZE>
void launch_moe(void const *input_ptr,
                            void const *weight_ptr,
                            void const *residual_ptr,
                            void *output_ptr,
                            void const *expert_routing,
                            void const *expert_mask) {
  constexpr int grid_x = expert_stride;
  constexpr int grid_y = 24;
  dim3 grid_dim(grid_x, grid_y, 1);
  dim3 block_dim(128, 1, 1);
  size_t smem_size = 96 * 1024;

  constexpr int output_size = OUTPUT_SIZE / grid_y;
  cudaFuncSetAttribute(
      moe_kernel_wrapper<T, BATCH_SIZE, output_size, OUTPUT_SIZE, REDUCTION_SIZE>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      smem_size);

  moe_kernel_wrapper<T, BATCH_SIZE, output_size, OUTPUT_SIZE, REDUCTION_SIZE>
      <<<grid_dim, block_dim, smem_size>>>(
          input_ptr, weight_ptr, residual_ptr, output_ptr, expert_routing, expert_mask);
}

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while (0)

int main(int argc, char** argv) {
    int iters = 500;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--iters") && i + 1 < argc) {
            iters = std::atoi(argv[++i]);
        }
    }

    unsigned int seed = 42;  // any constant number works
    std::mt19937 rng(seed);  // Mersenne Twister generator

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    // Host memory allocation
    std::vector<bfloat16> h_matrix_A(m * k);
    std::vector<bfloat16> h_matrix_B(experts_size * k * n);
    std::vector<bfloat16> h_residual(experts_size * m * n);
    std::vector<bfloat16> h_output_matrix(activate_experts_size * m * n);

    // Initialize host data (Example with dummy values)
    // for (size_t i = 0; i < h_matrix_A.size(); ++i) {
    //     h_matrix_A[i] = __float2bfloat16(1.0f); 
    // }
    // for (size_t i = 0; i < h_matrix_B.size(); ++i) {
    //     h_matrix_B[i] = __float2bfloat16(2.0f);
    // }
    // for (size_t i = 0; i < h_residual.size(); ++i) {
    //     h_residual[i] = __float2bfloat16(0.0f);
    // }


    for (int mm = 0; mm < m; mm ++) {
      for (int kk = 0; kk < k; kk ++) {
        // h_matrix_A[mm*k + kk] = __float2bfloat16(mm);
        h_matrix_A[mm*k + kk] = __float2bfloat16(1);
      }
    }

    for (int e = 0; e < experts_size; e ++) {
      for (int nn = 0; nn < n; nn ++) {
        for (int kk = 0; kk < k; kk ++) {
          h_matrix_B[e*n*k + nn*k + kk] = __float2bfloat16(e);
          // if(e == 95 && nn == 0 && kk == 0) {
          //   printf("expert 95 initialized as: %f\n", float(h_matrix_B[e*n*k + nn*k + kk]));
          // }
        }
      }
    }
    for (size_t i = 0; i < h_residual.size(); ++i) {
        h_residual[i] = __float2bfloat16(0.0f);
    }

    // For MOE related data
    std::vector<float> h_expert_score(m * experts_size);
    // 0 in route means not activate, other postivate number means the index of activate expert for this token.
    std::vector<uint32_t> h_expert_routing(experts_size * m);
    // whether the expert is activte for any token, it will be 0 if it's not activate, the last one is how many expert is activate.
    std::vector<uint32_t> h_expert_mask(experts_size + 1);

    for (size_t i = 0; i < h_expert_score.size(); ++i) {
        h_expert_score[i] = dist(rng); 
    }

    // Track which experts get activated at least once
    std::unordered_set<uint32_t> active_experts;

    for (size_t token_idx = 0; token_idx < m; ++token_idx) {
        // Each token’s slice of scores
        const float* token_scores = h_expert_score.data() + token_idx * experts_size;

        // Indices [0..experts_size)
        std::vector<uint32_t> indices(experts_size);
        std::iota(indices.begin(), indices.end(), 0);

        // Sort descending by score
        std::partial_sort(
            indices.begin(),
            indices.begin() + activate_experts_size,
            indices.end(),
            [&](uint32_t a, uint32_t b) { return token_scores[a] > token_scores[b]; });

        // Mark top-k as active experts for this token
        for (size_t k = 0; k < activate_experts_size; ++k) {
            uint32_t expert_id = indices[k];
            // store 1-based expert index (0 means not active)
            h_expert_routing[expert_id * m + token_idx] = k + 1;
            active_experts.insert(expert_id);
        }
    }

    // Fill mask
    int exp_idx = 0;
    for (uint32_t e : active_experts) {
      h_expert_mask[exp_idx ++] = e;
    }
    h_expert_mask.back() = static_cast<uint32_t>(active_experts.size());
    
    printf("total activae experts: %ld\n", active_experts.size());\
    printf("h_expert_routing(omit no activaed expert):\n");
    for(int i = 0; i < experts_size; i ++) {
      if(active_experts.count(i) == 0) {
        continue;
      }
      printf("expert %d: ", i);
      for(int j = 0; j < m; j ++) {
        printf("%d, ", h_expert_routing[i * m + j]);
      }
      printf("\n");
    }
    printf("\n");

    printf("h_expert_mask:\n");
    for(int i = 0; i < h_expert_mask.size(); i ++) {
      printf("%d, ", h_expert_mask[i]);
    }
    printf("\n");
    

    // Device memory allocation
    void* d_matrix_A = nullptr;
    void* d_matrix_B = nullptr;
    void* d_residual = nullptr;
    void* d_output_matrix = nullptr;

    void* d_expert_routing = nullptr;
    void* d_expert_mask = nullptr;

    CUDA_CHECK(cudaMalloc(&d_matrix_A, h_matrix_A.size() * sizeof(bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_matrix_B, h_matrix_B.size() * sizeof(bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_residual, h_residual.size() * sizeof(bfloat16)));
    CUDA_CHECK(cudaMalloc(&d_output_matrix, h_output_matrix.size() * sizeof(bfloat16)));

    CUDA_CHECK(cudaMalloc(&d_expert_routing, h_expert_routing.size() * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_expert_mask, h_expert_mask.size() * sizeof(uint32_t)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_matrix_A, h_matrix_A.data(), h_matrix_A.size() * sizeof(bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_matrix_B, h_matrix_B.data(), h_matrix_B.size() * sizeof(bfloat16), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_residual, h_residual.data(), h_residual.size() * sizeof(bfloat16), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(d_expert_routing, h_expert_routing.data(), h_expert_routing.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_expert_mask, h_expert_mask.data(), h_expert_mask.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaDeviceSynchronize());
    // Warmup
#if WARM_UP
    for (int i = 0; i < 10; ++i) {
        launch_moe<bfloat16, batch_size, n, k>(d_matrix_A, d_matrix_B, d_residual, d_output_matrix, d_expert_routing, d_expert_mask);
    }
#endif
    CUDA_CHECK(cudaDeviceSynchronize());

    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; ++i) {
        launch_moe<bfloat16, batch_size, n, k>(d_matrix_A, d_matrix_B, d_residual, d_output_matrix, d_expert_routing, d_expert_mask);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    float avg_ms = total_ms / iters;

    // Compute GFLOPs + bandwidth
    double flops = 2.0 * double(m) * double(n) * double(k) * active_experts.size();
    double gflops = (flops / (avg_ms * 1e-3)) / 1e9;

    double bytesA = double(m) * k * sizeof(bfloat16);
    double bytesB = double(k) * active_experts.size() * n * sizeof(bfloat16);
    double bytesD = double(m) * n * sizeof(bfloat16);
    double gbps = (bytesA + bytesB + bytesD) / (avg_ms * 1e-3) / 1e9;

    // Copy back result
    CUDA_CHECK(cudaMemcpy(h_output_matrix.data(), d_output_matrix, h_output_matrix.size()*sizeof(bfloat16), cudaMemcpyDeviceToHost));

    // Log (same as CUTLASS version)
    std::cout << "============== Baseline =================" << std::endl;
    std::cout << "BF16 GEMM -> FP32 output (M,N,K)=("
              << m << "," << n << "," << k << ")\n"
              << "iters=" << iters << "\n"
              << "Avg time: " << avg_ms << " ms\n"
              << "Perf: " << gflops << " GFLOP/s\n"
              << "BW:   " << gbps  << " GB/s\n";

    std::cout << "D[0..3]: ";
    for (int i = 0; i < std::min(n, 4); ++i) {
        std::cout << __bfloat162float(h_output_matrix[i]) << (i+1<4 ? ", " : "\n");
    }

    
    printf("Some outputs for first activated expert in tokens:\n");
    for(int i = 0; i < m; i ++) {
      printf("token %d: ", i);
      for(int j = 0; j < 4; j ++) {
        printf("%.4f, ", __bfloat162float(h_output_matrix[i*n*activate_experts_size + j]));
      }
      printf("\n");
    }

    // Free device memory
    CUDA_CHECK(cudaFree(d_matrix_A));
    CUDA_CHECK(cudaFree(d_matrix_B));
    CUDA_CHECK(cudaFree(d_residual));
    CUDA_CHECK(cudaFree(d_output_matrix));

    std::cout << "CUDA memory allocation and data transfer complete." << std::endl;
    return 0;
}