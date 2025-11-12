/* Copyright 2023-2024 CMU
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

/*
 * This file contains the SamplingFromLogitsKernel implementation.
 * Based on FlashInfer's sampling kernel (Apache License 2.0).
 */

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

#include <cub/cub.cuh>
#include <cuda/std/limits>
#include <numeric>
#include <utility>

namespace mirage {
namespace kernel {

using namespace cub;

// Define reduction operators based on CUDA version
#if CUDA_VERSION >= 12090
using MaxReduceOp = cuda::maximum<>;
#else
using MaxReduceOp = cub::Max;
#endif

// Block algorithm configurations
constexpr BlockScanAlgorithm SCAN_ALGO = BLOCK_SCAN_WARP_SCANS;
constexpr BlockReduceAlgorithm REDUCE_ALGO = BLOCK_REDUCE_WARP_REDUCTIONS;

// Helper function for ceiling division
template <typename T>
__host__ __device__ __forceinline__ T ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

/******************* vec_t - Simplified Vector Type *******************/

template <typename T, size_t vec_size>
struct vec_t {
  T data[vec_size];

  __device__ __forceinline__ T& operator[](size_t i) { return data[i]; }
  __device__ __forceinline__ const T& operator[](size_t i) const { return data[i]; }

  __device__ __forceinline__ void fill(T val) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      data[i] = val;
    }
  }

  __device__ __forceinline__ void cast_load(const T* ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      data[i] = ptr[i];
    }
  }
};

/******************* DataAndIndex Structure *******************/

template <typename DType, typename IdType>
struct DataAndIndex {
  DType data;
  IdType index;

  __device__ DataAndIndex operator+(const DataAndIndex& other) const {
    if (data > other.data) {
      return {data, index};
    } else {
      return {other.data, other.index};
    }
  }

  __device__ DataAndIndex& operator+=(const DataAndIndex& other) {
    if (data > other.data) {
      return *this;
    } else {
      data = other.data;
      index = other.index;
      return *this;
    }
  }
};

/******************* Gumbel Noise Generation *******************/

template <typename DType, uint32_t VEC_SIZE>
__device__ __forceinline__ vec_t<DType, VEC_SIZE> GenerateGumbelNoise(
    uint64_t philox_seed,
    uint64_t philox_offset,
    uint64_t subsequence) {
  curandStatePhilox4_32_10_t state;
  vec_t<float, VEC_SIZE> noise;
  constexpr float kEPSILON = 1e-20f;
  constexpr float kLOG2 = 0.6931471806f;

  auto uniform2gumbel = [](float x) {
    return -kLOG2 * log2f(-log2f(x + kEPSILON) + kEPSILON);
  };

#pragma unroll
  for (uint32_t i = 0; i + 4 <= VEC_SIZE; i += 4) {
    curand_init(philox_seed, subsequence + i, philox_offset, &state);
    float4 noise_vec = curand_uniform4(&state);
    noise[i] = uniform2gumbel(noise_vec.x);
    noise[i + 1] = uniform2gumbel(noise_vec.y);
    noise[i + 2] = uniform2gumbel(noise_vec.z);
    noise[i + 3] = uniform2gumbel(noise_vec.w);
  }

  if constexpr (VEC_SIZE % 4 != 0) {
    curand_init(philox_seed, subsequence + VEC_SIZE / 4 * 4, philox_offset, &state);
    float4 noise_vec = curand_uniform4(&state);
    if constexpr (VEC_SIZE % 4 == 1) {
      noise[VEC_SIZE - 1] = uniform2gumbel(noise_vec.x);
    } else if constexpr (VEC_SIZE % 4 == 2) {
      noise[VEC_SIZE - 2] = uniform2gumbel(noise_vec.x);
      noise[VEC_SIZE - 1] = uniform2gumbel(noise_vec.y);
    } else if constexpr (VEC_SIZE % 4 == 3) {
      noise[VEC_SIZE - 3] = uniform2gumbel(noise_vec.x);
      noise[VEC_SIZE - 2] = uniform2gumbel(noise_vec.y);
      noise[VEC_SIZE - 1] = uniform2gumbel(noise_vec.z);
    }
  }

  if constexpr (std::is_same_v<DType, float>) {
    return noise;
  } else {
    vec_t<DType, VEC_SIZE> ret;
#pragma unroll
    for (uint32_t i = 0; i < VEC_SIZE; ++i) {
      ret[i] = static_cast<DType>(noise[i]);
    }
    return ret;
  }
}

/******************* Sampling From Logits Kernel *******************/

/*!
 * \brief Sampling kernel that selects tokens from logits using Gumbel-Max trick.
 *
 * This kernel implements deterministic sampling from logits by:
 * 1. Adding Gumbel noise to each logit value
 * 2. Finding the index with maximum (logit + noise)
 *
 * The Gumbel-Max trick provides a differentiable approximation to sampling
 * and ensures reproducible results given the same random seed.
 *
 * \tparam BLOCK_THREADS Number of threads per block
 * \tparam SCAN_ALGORITHM CUB block scan algorithm
 * \tparam REDUCE_ALGORITHM CUB block reduce algorithm
 * \tparam VEC_SIZE Vector size for coalesced memory access
 * \tparam DETERMINISTIC Whether to use deterministic operations
 * \tparam DType Data type (e.g., float, half)
 * \tparam IdType Index type (e.g., int, uint32_t)
 *
 * \param logits Input logits tensor [batch_size, d]
 * \param output Output sampled indices [batch_size]
 * \param indices Optional batch indices to process [num_indices], or nullptr for sequential
 * \param d Vocabulary size (last dimension of logits)
 * \param philox_seed Random seed for Philox RNG
 * \param philox_offset Offset for Philox RNG
 */
template <uint32_t BLOCK_THREADS,
          BlockScanAlgorithm SCAN_ALGORITHM,
          BlockReduceAlgorithm REDUCE_ALGORITHM,
          uint32_t VEC_SIZE,
          bool DETERMINISTIC,
          typename DType,
          typename IdType>
__global__ void SamplingFromLogitsKernel(DType* logits,
                                         IdType* output,
                                         IdType* indices,
                                         uint32_t d,
                                         uint64_t philox_seed,
                                         uint64_t philox_offset) {
  const uint32_t bx = blockIdx.x, tx = threadIdx.x;
  const uint32_t row_idx = indices == nullptr ? bx : indices[bx];

  using SharedMem = typename BlockReduce<DataAndIndex<DType, IdType>,
                                         BLOCK_THREADS,
                                         REDUCE_ALGORITHM>::TempStorage;
  extern __shared__ __align__(alignof(SharedMem)) uint8_t smem_sampling_logit[];
  auto& temp_storage = reinterpret_cast<SharedMem&>(smem_sampling_logit);

  vec_t<DType, VEC_SIZE> logits_vec;
  DataAndIndex<DType, IdType> max_data = {
      -cuda::std::numeric_limits<DType>::infinity(), 0};

  // Process logits in chunks with vectorized loads
  for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i) {
    logits_vec.fill(-cuda::std::numeric_limits<DType>::infinity());

    // Load logits vector if within bounds
    if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
      logits_vec.cast_load(logits + row_idx * d + i * BLOCK_THREADS * VEC_SIZE +
                           tx * VEC_SIZE);
    }

    // Generate Gumbel noise
    vec_t<DType, VEC_SIZE> gumbel_noise = GenerateGumbelNoise<DType, VEC_SIZE>(
        philox_seed,
        philox_offset,
        static_cast<uint64_t>(bx * d + (i * BLOCK_THREADS + tx) * VEC_SIZE));

    // Add noise to logits and prepare for reduction
    DataAndIndex<DType, IdType> cur_data[VEC_SIZE];
#pragma unroll
    for (uint32_t j = 0; j < VEC_SIZE; ++j) {
      cur_data[j].data = (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d
                             ? logits_vec[j] + gumbel_noise[j]
                             : -cuda::std::numeric_limits<DType>::infinity();
      cur_data[j].index = (i * BLOCK_THREADS + tx) * VEC_SIZE + j;
    }

    // Find maximum across block
    max_data +=
        BlockReduce<DataAndIndex<DType, IdType>, BLOCK_THREADS, REDUCE_ALGORITHM>(
            temp_storage)
            .template Sum<VEC_SIZE>(cur_data);
  }

  // Write output
  if (tx == 0) {
    output[bx] = max_data.index;
  }
}

/******************* Dispatch Macros *******************/

#define DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, ...) \
  if (deterministic) {                                            \
    constexpr bool DETERMINISTIC = true;                          \
    __VA_ARGS__                                                   \
  } else {                                                        \
    constexpr bool DETERMINISTIC = false;                         \
    __VA_ARGS__                                                   \
  }

#define DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, ...) \
  if (compute_capacity.first >= 8) {                                           \
    constexpr uint32_t BLOCK_THREADS = 1024;                                   \
    __VA_ARGS__                                                                \
  } else {                                                                     \
    constexpr uint32_t BLOCK_THREADS = 512;                                    \
    __VA_ARGS__                                                                \
  }

#define DISPATCH_ALIGNED_VEC_SIZE(aligned_vec_size, ALIGNED_VEC_SIZE, ...) \
  switch (aligned_vec_size) {                                              \
    case 16: {                                                             \
      constexpr size_t ALIGNED_VEC_SIZE = 16;                              \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 8: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 8;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 4: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 4;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 2: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 2;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    case 1: {                                                              \
      constexpr size_t ALIGNED_VEC_SIZE = 1;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
    default: {                                                             \
      constexpr size_t ALIGNED_VEC_SIZE = 1;                               \
      __VA_ARGS__                                                          \
      break;                                                               \
    }                                                                      \
  }

/******************* Helper Functions *******************/

inline std::pair<int, int> GetCudaComputeCapability() {
  int device_id = 0;
  cudaGetDevice(&device_id);
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);
  return std::make_pair(major, minor);
}

/******************* Sampling From Logits Wrapper *******************/

/*!
 * \brief Sample tokens from logits using Gumbel-Max trick.
 *
 * This function provides a high-level interface to the SamplingFromLogitsKernel.
 * It automatically determines optimal launch parameters based on:
 * - GPU compute capability (for block size)
 * - Data type and vocabulary size (for vectorization)
 *
 * \tparam T Data type (float, half, etc.)
 * \tparam IdType Index type (int, uint32_t, etc.)
 *
 * \param logits Input logits tensor [batch_size, d]
 * \param output Output sampled token indices [batch_size]
 * \param indices Optional batch indices to process [num_indices], or nullptr for all batches
 * \param batch_size Number of batches to process
 * \param d Vocabulary size (last dimension of logits)
 * \param deterministic Whether to use deterministic operations (for debugging)
 * \param philox_seed Random seed for Philox RNG
 * \param philox_offset Offset for Philox RNG (for different sampling stages)
 * \param stream CUDA stream to launch kernel on (default: 0)
 *
 * \return cudaSuccess on success, error code otherwise
 *
 * Example usage:
 * \code
 * float* logits;        // [32, 50257]
 * int* output;          // [32]
 * uint32_t batch_size = 32;
 * uint32_t vocab_size = 50257;
 * uint64_t seed = 42;
 *
 * cudaError_t status = SamplingFromLogits(
 *     logits, output, nullptr, batch_size, vocab_size,
 *     false, seed, 0, 0);
 * \endcode
 */
template <typename T, typename IdType>
cudaError_t SamplingFromLogits(T* logits,
                                IdType* output,
                                IdType* indices,
                                uint32_t batch_size,
                                uint32_t d,
                                bool deterministic,
                                uint64_t philox_seed,
                                uint64_t philox_offset,
                                cudaStream_t stream = 0) {
  // Compute optimal vector size based on data type alignment and vocabulary size
  const uint32_t vec_size = std::gcd(16 / sizeof(T), d);

  auto compute_capacity = GetCudaComputeCapability();
  DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
    dim3 nblks(batch_size);
    dim3 nthrs(BLOCK_THREADS);
    void* args[] = {&logits, &output, &indices, &d, &philox_seed, &philox_offset};
    const uint32_t smem_size = sizeof(
        typename BlockReduce<DataAndIndex<T, IdType>, BLOCK_THREADS, REDUCE_ALGO>::TempStorage);

    DISPATCH_ALIGNED_VEC_SIZE(
        vec_size, VEC_SIZE, {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
          auto kernel = SamplingFromLogitsKernel<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE,
                                                 DETERMINISTIC, T, IdType>;
          cudaError_t status = cudaLaunchKernel((void*)kernel, nblks, nthrs, args, smem_size, stream);
          if (status != cudaSuccess) {
            return status;
          }
        })});
    return cudaSuccess;
  });
}

} // namespace kernel
} // namespace mirage
