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
size_t const MAX_SMEM_SIZE = 96 * 1024; // 96 KB
// Note that we actually save stensors' fingerprints on GPU device memory
// so MAX_SMEM_FP_SIZE can be larger than MAX_SMEM_SIZE
size_t const MAX_SMEM_FP_SIZE = 1024 * 1024;                      // 1 MB
size_t const MAX_DMEM_DATA_SIZE = (size_t)2 * 1024 * 1024 * 1024; // 2 GB
size_t const MAX_DMEM_FP_SIZE = (size_t)2 * 1024 * 1024 * 1024;   // 2 GB
size_t const MAX_NUM_THREADBLOCKS_PER_KERNEL = 1024;
int const MAX_NUM_GPUS = 16;
int const DEFAULT_TB_REDUCTION_DIMX = 64;
int const MAX_NUM_WARP_GROUPS = 4;
int const NUM_THREADS_PER_WARP = 32;
int const NUM_WARPS_PER_GROUP = 4;
int const NUM_THREADS_PER_GROUP = NUM_WARPS_PER_GROUP * NUM_THREADS_PER_WARP;
} // namespace config
} // namespace mirage
