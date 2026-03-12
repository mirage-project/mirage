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

#pragma once
#include <cstddef>
#include <cstdint>

namespace mirage {
namespace config {

uint16_t const FP_P = 167;
uint16_t const FP_Q = 83;
uint32_t const FP_EXP_BASE = 3;
uint16_t const FP_PQ = 13861;
// FP_P_MUL_Q_MOD_1 is a multiplier of P and is 1 module Q
uint16_t const FP_P_MUL_Q_MOD_1 = 167;
// FP_Q_MUL_P_MOD_1 is a multiplier of Q and is 1 module P
uint16_t const FP_Q_MUL_P_MOD_1 = 13695;
size_t const MAX_NUM_THREADBLOCKS_PER_KERNEL = 4096;
int const MAX_NUM_DEVICES = 16;
constexpr int MAX_TENSOR_DIMS = 4;
int const DEFAULT_TB_REDUCTION_DIMX = 64;
int const MAX_NUM_WARP_GROUPS = 4;
int const NUM_THREADS_PER_WARP = 32;
int const NUM_WARPS_PER_GROUP = 4;
int const NUM_THREADS_PER_GROUP = NUM_WARPS_PER_GROUP * NUM_THREADS_PER_WARP;
constexpr int MAX_TMA_DESC_PER_TENSOR = 3;

#if defined(MIRAGE_BACKEND_USE_CUDA) && defined(MIRAGE_BACKEND_USE_NKI)
#error                                                                         \
    "Both MIRAGE_BACKEND_USE_CUDA and MIRAGE_BACKEND_USE_NKI are defined. Please define only one backend type."
#elif defined(MIRAGE_BACKEND_USE_CUDA)
size_t const MAX_DMEM_SIZE = (size_t)2 * 1024 * 1024 * 1024;    // 2 GB
size_t const MAX_SMEM_SIZE = 96 * 1024;                         // 96 KB
#elif defined(MIRAGE_BACKEND_USE_NKI)
size_t const MAX_DMEM_SIZE = (size_t)32 * 1024 * 1024 * 1024;    // 32 GB
size_t const MAX_SMEM_SIZE = (size_t)24 * 1024 * 1024;           // 24 MB
#else
#error "Please define either MIRAGE_BACKEND_USE_CUDA or MIRAGE_BACKEND_USE_NKI."
#endif

// Note that we actually save stensors' fingerprints on GPU device memory
// so MAX_SMEM_FP_SIZE can be larger than MAX_SMEM_SIZE
#if defined(MIRAGE_FINGERPRINT_USE_CUDA) && defined(MIRAGE_FINGERPRINT_USE_CPU)
#error                                                                         \
    "Both MIRAGE_FINGERPRINT_USE_CUDA and MIRAGE_FINGERPRINT_USE_CPU are defined. Please define only one fingerprint type."
#elif defined(MIRAGE_FINGERPRINT_USE_CUDA)
size_t const MAX_DMEM_FP_SIZE = (size_t)2 * 1024 * 1024 * 1024; // 2 GB
size_t const MAX_SMEM_FP_SIZE = (size_t)1024 * 1024;            // 1 MB
#else
size_t const MAX_DMEM_FP_SIZE = (size_t)64 * 1024 * 1024 * 1024; // 64 GB
size_t const MAX_SMEM_FP_SIZE = (size_t)64 * 1024 * 1024;        // 64 MB
#endif

} // namespace config
} // namespace mirage
