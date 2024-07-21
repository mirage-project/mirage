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
#include <vector>

#include "mirage/kernel/device_tensor.h"
#include "mirage/threadblock/smem_tensor.h"

namespace mirage {
namespace transpiler {

// Metadata for one DTensor during transpiling
// DTensors with the same `guid` share one DTensorMeta
struct DTensorMeta {
  bool is_input; // Whether this tensor is an input tensor
  int input_idx; // Index among all input tensors

  bool is_output; // Whether this tensor is an output tensor
  int output_idx; // Index among all output tensors

  // Stride of each dimension, in number of elements
  // We may pad the strides to multiple of 8 to leverage `cp.async` instruction
  // Refer to this discussion for more information about the layout of
  // DTensor/STensor: https://github.com/mirage-project/mirage/discussions/33
  size_t strides[kernel::MAX_TENSOR_DIMS];

  // Innermost dimension (dimension with stride=1)
  int innermost_dim;

  // The start address in the global memory, in bytes
  // This may not be the same as DTensor::data_offset since
  // - 1. The layout has changed because of paddings
  // - 2. We do not need to allocate fingerprints
  // - 3. We want to use more advanced allocation strategies
  // This addr must be aligned to 16 bytes for intermediate tensors, but
  // may not be aligned for input tensors
  // TODO(intlsy) Allow unaligned input tensors
  size_t addr;
};

// Metadata for STensors during transpiling
// STensors with the same `guid` share one STensorMeta
struct STensorMeta {
  // Innermost dimension (dimension with stride=1)
  int innermost_dim;

  // Dimensions that are swizzled
  std::vector<int> swizzled_dims;

  // The start address in the shared memory, in bytes
  size_t addr;

  // Strides of each dimension, in number of elements
  // Must be padded to multiple of 8
  size_t strides[threadblock::MAX_TENSOR_DIMS];
};

} // namespace transpiler
} // namespace mirage