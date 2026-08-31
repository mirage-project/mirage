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
#include "mirage/transpiler/error_types.h"

namespace mirage {
namespace transpiler {

// Configuration for the transpiler
struct TranspilerConfig {
  // Target compute capability, should be compute_capability*10, e.g. A100 is 80
  // and H100 is 90
  int target_cc;

  bool profiling = false;

  // features for GPUs >= Grace Hopper
  //
  // These MUST have defaults: callers construct TranspilerConfig as a bare
  // stack value (e.g. `cdef TranspilerConfig` in core.pyx) and only assign the
  // warp-group fields when the caller supplied both num_warp_groups and
  // pipeline_stages. Without defaults the Hopper/Blackwell backends read
  // garbage and reject the graph with an empty `code` string, which surfaces in
  // Python as an inscrutable "CUDA compilation error".
  //
  // The defaults mirror num_warp_groups=2: one producer + one consumer.
  int num_consumer_wgs = 1;
  int num_producer_wgs = 1;
  int pipeline_stages = 2;

  // Pad the MMA's N mode up to MMA_N_MIN_PADDED. tcgen05 accepts any N that is
  // a multiple of 8 up to 256, so an MPK decode batch -- M = 1..8 tokens, which
  // swapAB puts in the MMA's N -- is legal exactly as it is. Legal is not
  // necessarily fast: the hand-written linear_sm100 pads the same 8 tokens to
  // MMA_N = 16, spending half the accumulator on rows it discards to get a
  // wider instruction. Whether that trade pays for a GENERATED body is a
  // measurement, so it is a flag rather than a rule. Default off: the default
  // build must be unchanged.
  bool pad_mma_n = false;

  // Whether to enable graph rewriting
  bool enable_online_softmax = false;

  // Emit the custom op as a `__device__ __forceinline__` function taking the
  // shared-memory block as a parameter, instead of a `__global__` kernel that
  // declares `extern __shared__`. This is the form an MPK megakernel task body
  // needs: the megakernel owns the launch and the smem allocation, and calls
  // the generated function from its task dispatch.
  //
  // Only valid when the op needs no TMA descriptors -- those are built on the
  // host and passed as kernel arguments, which a task body has no way to
  // receive. The Blackwell backend rejects the combination rather than emitting
  // a function that references undeclared descriptors.
  bool emit_device_body = false;
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
  // A state indicating whether the transpile kernel is
  // valid or not
  TranspileErrorType error_type;

  // The generated CUDA code
  std::string code;

  // The size of the buffer (should be an array on GPU), in bytes
  size_t buf_size;

  // The maximum smem size used by a kernel, in bytes
  size_t max_smem_size;

  size_t profiler_buf_size;

  // Directives for output tensors
  std::vector<OutputTensorDirective> output_directives;
};

struct TiledMMA {
  std::string A_type;
  std::string B_type;
  std::string C_type;
  int M_tile_size;
  int N_tile_size;
  int K_tile_size;
  size_t guid;
  // swapAB: the device side computes C^T = B^T * A^T when M is not a legal
  // 1-SM tcgen05 M-tile (64/128), e.g. decode-shaped M = 1..8. M/N_tile_size
  // stay the *logical* m,n; consumers swap them together with the Major flags.
  // The host side in transpiler_kn.cc emits tiled_mma_<guid> for TMA atom
  // construction under the SAME symbol name as the device side, so the two must
  // agree -- when they did not, an M=8 K-loop failed with "SM100_MMA_F16BF16
  // M-mode size should be 64 or 128".
  bool swap_ab = false;

  // Constructor
  TiledMMA()
      : A_type("half_t"), B_type("half_t"), C_type("half_t"), M_tile_size(256),
        N_tile_size(256), K_tile_size(16), guid(0) {}

  TiledMMA(std::string const &A_type,
           std::string const &B_type,
           std::string const &C_type,
           int M_tile_size,
           int N_tile_size,
           int K_tile_size,
           size_t guid,
           bool swap_ab = false)
      : A_type(A_type), B_type(B_type), C_type(C_type),
        M_tile_size(M_tile_size), N_tile_size(N_tile_size),
        K_tile_size(K_tile_size), guid(guid), swap_ab(swap_ab) {}
};

struct TMAParams {
  size_t input_id; // Index among the GRAPH's inputs (dtensor_meta.input_idx)
  // Index among the OWNING OP's operands. This is what a megakernel task
  // indexes its input_ptrs[] / input_tma_desc_ptrs[] by, and it differs from
  // input_id as soon as a graph holds more than one customized op.
  size_t operand_id = 0;
  size_t guid;
  size_t sguid;
  std::string srcLayout; // String representing the layout
  std::string dstLayout; // String representing the layout
  std::string tile_size; // String representing the tile
  bool m_input;
  std::tuple<int, int, int> clusterSize; // Tuple for cluster size
  std::vector<int> original_shape;
  std::vector<size_t> original_stride;
  std::vector<int> partition_logic;
  int forloop_range;
  int forloop_dim;
  std::string multicast_direction;
  TiledMMA tiled_mma;

  // Constructor for convenience
  TMAParams(size_t input_id,
            size_t guid,
            size_t sguid,
            std::string const &srcLayout,
            std::string const &dstLayout,
            bool const m_input,
            std::string const &tile_size,
            std::tuple<int, int, int> const &clusterSize,
            std::vector<int> const &original_shape,
            std::vector<size_t> const &original_stride,
            std::vector<int> const &partition_logic,
            int forloop_range,
            int forloop_dim,
            std::string const &multicast_direction = "NOT_MULTICAST",
            TiledMMA const &tiled_mma =
                TiledMMA("half_t", "half_t", "half_t", 256, 256, 16, 0))
      : input_id(input_id), guid(guid), sguid(sguid), srcLayout(srcLayout),
        dstLayout(dstLayout), m_input(m_input), tile_size(tile_size),
        clusterSize(clusterSize), original_shape(original_shape),
        original_stride(original_stride), partition_logic(partition_logic),
        forloop_range(forloop_range), forloop_dim(forloop_dim),
        multicast_direction(multicast_direction), tiled_mma(tiled_mma) {}
};

// Transpile a custom KN operator (a custom block graph)
struct CustomOPTranspileResult {
  // A state indicating whether the transpile kernel is
  // valid or not
  TranspileErrorType error_type;

  // The name of the generated kernel function
  std::string func_name;
  // The size of the shared memory, in bytes
  size_t smem_size;

  size_t profiler_buf_size;
  // The kernel function code. Should be something like:
  // __global__ void <func_name>(InputDTensor0, ..., InputDTensorN) {
  //  [kernel code]
  // }
  std::string code;
  std::vector<TMAParams> tmaParamsList;
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
  size_t strides[mirage::config::MAX_TENSOR_DIMS];

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
// The MMA N mode a padded task uses. Kept next to STensorMeta because TWO
// passes must agree on it: transpiler_tb_blackwell emits the MMA shape, and
// resolve_tensor_layout sizes the smem buffer the MMA reads. If they disagree
// the UMMA reads past its operand into the neighbouring tensor -- wrong
// numbers, not a crash.
constexpr int MMA_N_MIN_PADDED = 16;

inline int padded_mma_n(int mma_n, bool enabled) {
  return (enabled && mma_n > 0 && mma_n < MMA_N_MIN_PADDED) ? MMA_N_MIN_PADDED
                                                            : mma_n;
}

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

  // Major K for M input
  bool m_input = false;

  // is N input for matmul
  bool n_input = false;

  // is the OUTPUT (C tile) of a matmul. c_matrix_guid is only set on the
  // operands, so the output needs its own mark -- config.pad_mma_n has to
  // widen the C tile too, since write_tC_to_sC lands mma_n rows in it.
  bool matmul_output = false;

  // the guid of the matrix tensor
  // if have a m matrix as pair with this tensor, which means this matrix is A
  // matrix m_matrix_guid is the guid of the pair tensor guid
  sguid_t m_matrix_guid = 0;
  // if have a n matrix as pair with this tensor, which means this matrix is B
  // matrix n_matrix_guid is the guid of the pair tensor guid
  sguid_t n_matrix_guid = 0;
  // if have a c matrix as output tensor, which means this matrix is C matrix
  // c_matrix_guid is the guid of the pair tensor guid
  sguid_t c_matrix_guid = 0;

  // if is associate with a tma/cp.async copy
  bool is_pipelined_input = false;

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

  // The guid offset for tmem base ptr used in blackwell
  sguid_t tmem_base_ptr_guid;

  // The guid offset for mbarrier ptr used in blackwell
  sguid_t mbarrier_buf_guid_offset;
};

} // namespace transpiler
} // namespace mirage
