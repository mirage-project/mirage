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

// CUTLASS and CuTe headers
#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"

// Include FlashInfer's CUTLASS MLA implementation
#include "flashinfer_mla/kernel/sm100_mla_tile_scheduler.hpp"
#include "flashinfer_mla/device/sm100_mla.hpp"

namespace mirage {
namespace mla {

using namespace cute;
using namespace cutlass;

// MLA kernel configuration for DeepSeek-style MLA
// - 128 heads (fixed)
// - Latent dimension: 512 (compressed KV)
// - Rope dimension: 64 (position embeddings)
// - Total head dim: 576 = 512 + 64

template <typename Element = cutlass::bfloat16_t,
          typename ElementAcc = float,
          typename ElementOut = cutlass::bfloat16_t,
          typename ElementLSE = float,
          int TileS = 128>  // Sequence tile size
struct MLAKernelConfig {
  // Fixed MLA dimensions for DeepSeek
  static constexpr int kNumHeads = 128;
  static constexpr int kHeadDimLatent = 512;
  static constexpr int kHeadDimRope = 64;
  static constexpr int kHeadDimTotal = kHeadDimLatent + kHeadDimRope;

  using TileShape = Shape<Int<kNumHeads>, Int<TileS>, Shape<Int<kHeadDimLatent>, Int<kHeadDimRope>>>;

  // Use persistent tile scheduler for better performance
  using TileScheduler = cutlass::fmha::kernel::Sm100MlaPersistentTileScheduler;

  using Kernel = cutlass::fmha::kernel::Sm100FmhaMlaKernelTmaWarpspecialized<
      TileShape, Element, ElementAcc, ElementOut, ElementLSE, TileScheduler>;

  using Device = cutlass::fmha::device::MLA<Kernel>;
};

// Helper struct for MLA arguments
template <typename Element>
struct MLAArguments {
  // Query tensors (after matrix absorption)
  Element* ptr_q_latent;      // [batch, num_heads, head_dim_latent]
  Element* ptr_q_rope;        // [batch, num_heads, head_dim_rope]

  // KV cache (compressed)
  Element* ptr_c_latent;      // [num_pages, page_size, head_dim_latent] or [batch, seq_len, head_dim_latent]
  Element* ptr_k_rope;        // [num_pages, page_size, head_dim_rope] or [batch, seq_len, head_dim_rope]

  // Output
  Element* ptr_output;        // [batch, num_heads, head_dim_latent]
  float* ptr_lse;             // [batch, num_heads] (optional)

  // Paged attention metadata (optional)
  int* ptr_seq_lens;          // [batch] - sequence lengths
  int* ptr_page_table;        // [batch, max_pages] - page indices
  int page_size;              // Page size (power of 2, <= 128)
  int page_count;             // Total number of pages

  // Problem dimensions
  int batch_size;
  int max_seq_len;

  // Softmax scale (typically 1/sqrt(head_dim_qk))
  float softmax_scale;

  // Split-K for long sequences (-1 for auto)
  int split_kv = -1;

  // Workspace
  void* workspace;
  size_t workspace_size;
};

// Launch MLA kernel
template <typename Config = MLAKernelConfig<>>
cudaError_t launch_mla_attention(
    MLAArguments<typename Config::Kernel::Element> const& args,
    cudaStream_t stream = nullptr) {

  using Device = typename Config::Device;
  using Kernel = typename Config::Kernel;
  using Element = typename Kernel::Element;
  using ElementAcc = typename Kernel::ElementAcc;
  using ElementOut = typename Kernel::ElementOut;
  using ElementLSE = typename Kernel::ElementLSE;

  // Build kernel arguments
  typename Kernel::Arguments kernel_args;

  // Problem shape: (num_heads=128, seqlen, (d_latent=512, d_rope=64), batch)
  kernel_args.problem_shape = make_shape(
      Int<Config::kNumHeads>{},
      args.max_seq_len,
      make_shape(Int<Config::kHeadDimLatent>{}, Int<Config::kHeadDimRope>{}),
      args.batch_size);

  // Mainloop arguments
  kernel_args.mainloop.softmax_scale = args.softmax_scale;
  kernel_args.mainloop.ptr_q_latent = args.ptr_q_latent;
  kernel_args.mainloop.stride_q_latent = make_tuple(
      static_cast<int64_t>(Config::kHeadDimLatent), _1{},
      static_cast<int64_t>(Config::kNumHeads * Config::kHeadDimLatent));
  kernel_args.mainloop.ptr_q_rope = args.ptr_q_rope;
  kernel_args.mainloop.stride_q_rope = make_tuple(
      static_cast<int64_t>(Config::kHeadDimRope), _1{},
      static_cast<int64_t>(Config::kNumHeads * Config::kHeadDimRope));

  if (args.ptr_page_table != nullptr) {
    // Paged attention mode
    kernel_args.mainloop.ptr_c_latent = args.ptr_c_latent;
    kernel_args.mainloop.stride_c_latent = make_tuple(
        static_cast<int64_t>(Config::kHeadDimLatent), _1{},
        static_cast<int64_t>(args.page_size * Config::kHeadDimLatent));
    kernel_args.mainloop.ptr_k_rope = args.ptr_k_rope;
    kernel_args.mainloop.stride_k_rope = make_tuple(
        static_cast<int64_t>(Config::kHeadDimRope), _1{},
        static_cast<int64_t>(args.page_size * Config::kHeadDimRope));

    kernel_args.mainloop.ptr_seq = args.ptr_seq_lens;
    kernel_args.mainloop.ptr_page_table = args.ptr_page_table;
    kernel_args.mainloop.stride_page_table = make_tuple(_1{}, args.batch_size);
    kernel_args.mainloop.page_count = args.page_count;
    kernel_args.mainloop.page_size = args.page_size;
  } else {
    // Non-paged mode (contiguous KV cache)
    kernel_args.mainloop.ptr_c_latent = args.ptr_c_latent;
    kernel_args.mainloop.stride_c_latent = make_tuple(
        static_cast<int64_t>(Config::kHeadDimLatent), _1{},
        static_cast<int64_t>(args.max_seq_len * Config::kHeadDimLatent));
    kernel_args.mainloop.ptr_k_rope = args.ptr_k_rope;
    kernel_args.mainloop.stride_k_rope = make_tuple(
        static_cast<int64_t>(Config::kHeadDimRope), _1{},
        static_cast<int64_t>(args.max_seq_len * Config::kHeadDimRope));
  }

  // Epilogue arguments
  kernel_args.epilogue.ptr_o = args.ptr_output;
  kernel_args.epilogue.stride_o = make_tuple(
      static_cast<int64_t>(Config::kHeadDimLatent), _1{},
      static_cast<int64_t>(Config::kNumHeads * Config::kHeadDimLatent));
  kernel_args.epilogue.ptr_lse = args.ptr_lse;
  kernel_args.epilogue.stride_lse = make_tuple(_1{}, Config::kNumHeads);

  // Hardware info
  int device_id;
  cudaGetDevice(&device_id);
  cudaDeviceProp props;
  cudaGetDeviceProperties(&props, device_id);
  kernel_args.hw_info.sm_count = props.multiProcessorCount;
  kernel_args.hw_info.device_id = device_id;

  // Split-K
  kernel_args.split_kv = args.split_kv;
  if (kernel_args.split_kv < 0) {
    Device::set_split_kv(kernel_args);
  }

  // Check if kernel can be launched
  auto status = Device::can_implement(kernel_args);
  if (status != Status::kSuccess) {
    return cudaErrorInvalidConfiguration;
  }

  // Get workspace size and verify
  size_t required_workspace = Device::get_workspace_size(kernel_args);
  if (args.workspace_size < required_workspace) {
    return cudaErrorInvalidValue;
  }

  // Launch kernel
  Device mla_device;
  status = mla_device.run(kernel_args, args.workspace, stream);

  return (status == Status::kSuccess) ? cudaSuccess : cudaErrorLaunchFailure;
}

// Get required workspace size
template <typename Config = MLAKernelConfig<>>
size_t get_mla_workspace_size(int batch_size, int max_seq_len, int split_kv = -1) {
  using Device = typename Config::Device;
  using Kernel = typename Config::Kernel;

  typename Kernel::Arguments args;
  args.problem_shape = make_shape(
      Int<Config::kNumHeads>{},
      max_seq_len,
      make_shape(Int<Config::kHeadDimLatent>{}, Int<Config::kHeadDimRope>{}),
      batch_size);
  args.split_kv = split_kv;

  if (args.split_kv < 0) {
    // Get SM count for auto split-kv calculation
    int device_id;
    cudaGetDevice(&device_id);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);
    args.hw_info.sm_count = props.multiProcessorCount;
    Device::set_split_kv(args);
  }

  return Device::get_workspace_size(args);
}

}  // namespace mla
}  // namespace mirage
