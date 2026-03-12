/* Copyright (c) 2025 by CMU.
 * Copyright (c) 2025 by FlashInfer team.
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
 * Sampling from logits using Gumbel-Max trick
 * Based on FlashInfer's sampling kernel (Apache License 2.0).
 */

#pragma once

#include <cub/cub.cuh>
#include <cuda.h>
#include <cuda/std/limits>
#include <curand.h>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>

namespace kernel {

using namespace cub;

// Helper function for ceiling division
template <typename T>
__host__ __device__ __forceinline__ T sampling_ceil_div(T a, T b) {
  return (a + b - 1) / b;
}

/******************* vec_t - Simplified Vector Type *******************/

template <typename T, size_t vec_size>
struct sampling_vec_t {
  T data[vec_size];

  __device__ __forceinline__ T &operator[](size_t i) {
    return data[i];
  }
  __device__ __forceinline__ T const &operator[](size_t i) const {
    return data[i];
  }

  __device__ __forceinline__ void fill(T val) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      data[i] = val;
    }
  }

  __device__ __forceinline__ void cast_load(T const *ptr) {
#pragma unroll
    for (size_t i = 0; i < vec_size; ++i) {
      data[i] = ptr[i];
    }
  }
};

/******************* DataAndIndex Structure *******************/

template <typename DType, typename IdType>
struct SamplingDataAndIndex {
  DType data;
  IdType index;

  __device__ SamplingDataAndIndex
      operator+(SamplingDataAndIndex const &other) const {
    if (data > other.data) {
      return {data, index};
    } else {
      return {other.data, other.index};
    }
  }

  __device__ SamplingDataAndIndex &
      operator+=(SamplingDataAndIndex const &other) {
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
__device__ __forceinline__ sampling_vec_t<DType, VEC_SIZE>
    GenerateSamplingGumbelNoise(uint64_t philox_seed,
                                uint64_t philox_offset,
                                uint64_t subsequence) {
  curandStatePhilox4_32_10_t state;
  sampling_vec_t<float, VEC_SIZE> noise;
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
    curand_init(
        philox_seed, subsequence + VEC_SIZE / 4 * 4, philox_offset, &state);
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
    sampling_vec_t<DType, VEC_SIZE> ret;
#pragma unroll
    for (uint32_t i = 0; i < VEC_SIZE; ++i) {
      ret[i] = static_cast<DType>(noise[i]);
    }
    return ret;
  }
}

/******************* Sampling From Logits Kernel *******************/

constexpr BlockScanAlgorithm SAMPLING_SCAN_ALGO = BLOCK_SCAN_WARP_SCANS;
constexpr BlockReduceAlgorithm SAMPLING_REDUCE_ALGO =
    BLOCK_REDUCE_WARP_REDUCTIONS;

template <uint32_t BLOCK_THREADS,
          uint32_t VEC_SIZE,
          typename DType,
          typename IdType>
__device__ __forceinline__ void
    sampling_from_logits_kernel(DType *logits,
                                IdType *output,
                                uint32_t d,
                                uint64_t philox_seed,
                                uint64_t philox_offset,
                                int batch_size) {
  const uint32_t tx = threadIdx.x;

  using SharedMem = typename BlockReduce<SamplingDataAndIndex<DType, IdType>,
                                         BLOCK_THREADS,
                                         SAMPLING_REDUCE_ALGO>::TempStorage;
  extern __shared__ __align__(alignof(SharedMem)) uint8_t smem_sampling_logit[];
  auto &temp_storage = reinterpret_cast<SharedMem &>(smem_sampling_logit);

  // Loop over all batches
  for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    sampling_vec_t<DType, VEC_SIZE> logits_vec;
    SamplingDataAndIndex<DType, IdType> max_data = {
        -cuda::std::numeric_limits<DType>::infinity(), 0};

    // Process logits in chunks with vectorized loads
    for (uint32_t i = 0; i < sampling_ceil_div(d, BLOCK_THREADS * VEC_SIZE);
         ++i) {
      logits_vec.fill(-cuda::std::numeric_limits<DType>::infinity());

      // Load logits vector if within bounds
      if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d) {
        logits_vec.cast_load(logits + batch_idx * d +
                             i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
      }

      // Generate Gumbel noise
      sampling_vec_t<DType, VEC_SIZE> gumbel_noise =
          GenerateSamplingGumbelNoise<DType, VEC_SIZE>(
              philox_seed,
              philox_offset,
              static_cast<uint64_t>(batch_idx * d +
                                    (i * BLOCK_THREADS + tx) * VEC_SIZE));

      // Add noise to logits and prepare for reduction
      SamplingDataAndIndex<DType, IdType> cur_data[VEC_SIZE];
#pragma unroll
      for (uint32_t j = 0; j < VEC_SIZE; ++j) {
        cur_data[j].data = (i * BLOCK_THREADS + tx) * VEC_SIZE + j < d
                               ? logits_vec[j] + gumbel_noise[j]
                               : -cuda::std::numeric_limits<DType>::infinity();
        cur_data[j].index = (i * BLOCK_THREADS + tx) * VEC_SIZE + j;
      }

      // Find maximum across block
      max_data += BlockReduce<SamplingDataAndIndex<DType, IdType>,
                              BLOCK_THREADS,
                              SAMPLING_REDUCE_ALGO>(temp_storage)
                      .template Sum<VEC_SIZE>(cur_data);
    }

    // Write output for this batch
    if (tx == 0) {
      output[batch_idx] = max_data.index;
    }

    // Sync before next batch iteration to reuse shared memory
    __syncthreads();
  }
}

} // namespace kernel
