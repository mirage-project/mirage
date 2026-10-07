/* Copyright 2025 CMU
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

#include <cstdint>

// OCP MXFP4 as stored in the GPT-OSS checkpoint.
//   element  : E2M1, two values per byte, low nibble = even K, high nibble = odd K
//   scale    : UE8M0, one byte per 32 K elements, decoded as 2^(byte - 127)
//   packing  : along K, 16 bytes per 32 elements. A row of K is K/2 contiguous bytes.

namespace kernel {
namespace mxfp4 {

constexpr int BLOCK = 32;

__host__ __device__ __forceinline__ float e2m1(int nibble) {
  // transformers.integrations.mxfp4.FP4_VALUES
  constexpr float lut[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                             -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f,
                             -6.0f};
  return lut[nibble & 15];
}

__host__ __device__ __forceinline__ float ue8m0(unsigned scale) {
  return exp2f(static_cast<float>(static_cast<int>(scale) - 127));
}

// blocks: K/2 bytes, low nibble at even K. scales: K/32 bytes.
__host__ __device__ __forceinline__ float
    dequant(const uint8_t *row_bytes, const uint8_t *row_scales, int k) {
  uint8_t byte = row_bytes[k >> 1];
  int nibble = (k & 1) ? (byte >> 4) : (byte & 0x0f);
  return e2m1(nibble) * ue8m0(row_scales[k >> 5]);
}

} // namespace mxfp4
} // namespace kernel
