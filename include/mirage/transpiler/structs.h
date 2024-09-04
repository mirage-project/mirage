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

#include <algorithm>
#include <cstddef>
#include <vector>

#include "mirage/kernel/device_tensor.h"
#include "mirage/threadblock/smem_tensor.h"
#include "mirage/transpiler/common.h"

namespace mirage {
namespace transpiler {

// Configuration for the transpiler
struct TranspilerConfig {
  // Target compute capability, should be compute_capability*10, e.g. A100 is 80
  // and H100 is 90
  int target_cc;
};

// Directive for an output tensor
// The Python interface is responsible for allocating the tensor and passing
// the obtained pointer to `execute_mugraph`. So during transpilation, we must
// tell the Python interface how to allocate the tensor
struct OutputTensorDirective {
  // The physical size needed (in number of elements)
  size_t alloc_size;

  // The shape
  std::vector<int> shape;

  // The stride
  std::vector<size_t> strides;
};

// Result returned by the transpiler
struct TranspileResult {
  // The generated CUDA code
  std::string code;

  // The size of the buffer (should be an array on GPU), in bytes
  size_t buf_size;

  // Directives for output tensors
  std::vector<OutputTensorDirective> output_directives;
};

// Transpile a custom KN operator (a custom block graph)
struct CustomOPTranspileResult {
  // The name of the generated kernel function
  std::string func_name;
  // The size of the shared memory, in bytes
  size_t smem_size;
  // The kernel function code. Should be something like:
  // __global__ void <func_name>(InputDTensor0, ..., InputDTensorN) {
  //  [kernel code]
  // }
  std::string code;
};

// Metadata for one DTensor during transpiling
// DTensors with the same `guid` share one DTensorMeta
struct DTensorMeta {
  bool is_input; // Whether this tensor is an input tensor
  int input_idx; // Index among all input tensors

  bool is_output; // Whether this tensor is an output tensor
  int output_idx; // Index among all output tensors

  // Stride of each dimension, in number of elements
  // We may pad the strides to multiple of 8 to leverage `cp.async` instruction
  // Refer to the document for more information about the layout of
  // DTensor/STensor
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
  size_t addr;

  // Physical size needed for the tensor
  // For output tensors, we will return it back to the Python interface so that
  // it knows how much memory to allocate
  // (in number of elements)
  size_t num_phy_elems;
};

// Metadata for STensors during transpiling
// STensors with the same `guid` share one STensorMeta
struct STensorMeta {
  // Innermost dimension (dimension with stride=1)
  int innermost_dim;

  // Dimension that is swizzled
  // Currently we only allow swizzling one dimension
  // -1 means no swizzling
  int swizzled_dim;

  // Strides of each dimension, in number of elements
  // Must be padded to multiple of 8
  size_t strides[threadblock::MAX_TENSOR_DIMS];

  // Physical size needed for the tensor (in number of elements)
  size_t num_phy_elems;

  // Whether this tensor needs to be XOR-based swizzled
  bool is_xor_swizzled;

  // XOR-based swizzling parameters
  int xor_swizzle_b, xor_swizzle_m, xor_swizzle_s;
};

struct TBMemoryPlan {
  // The start address of each STensor
  std::unordered_map<sguid_t, size_t> addrs;

  // The size of the shared memory buffer, in bytes
  size_t smem_size;

  // The guid offset of a async input buffer
  // (i.e. if the guid of the output STensor of a software pipelined input is
  // $x$, then the guid of the async input buffer is $x +
  // pipelined_input_buf_guid_offset$)
  sguid_t pipelined_input_buf_guid_offset;
};

} // namespace transpiler
} // namespace mirage