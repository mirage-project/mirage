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

#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <torch/extension.h>

// Include the Mirage MLA kernel
#include "mirage/persistent_kernel/tasks/blackwell/mla_sm100_2sm.cuh"

using namespace cute;
using namespace cutlass;
using namespace cutlass::fmha;

// MLA kernel configuration (DeepSeek-V2 style)
constexpr int NUM_HEADS = 128;
constexpr int HEAD_DIM_LATENT = 512;
constexpr int HEAD_DIM_ROPE = 64;
constexpr int HEAD_DIM_TOTAL = HEAD_DIM_LATENT + HEAD_DIM_ROPE;
constexpr int TILE_S = 128; // Sequence tile size

// Define the kernel type
// TileShape = (H=128, S=128, D=(L=512, R=64))
using TileShape = Shape<Int<NUM_HEADS>,
                        Int<TILE_S>,
                        Shape<Int<HEAD_DIM_LATENT>, Int<HEAD_DIM_ROPE>>>;
using Element = cutlass::bfloat16_t;
using ElementAcc = float;
using ElementOut = cutlass::bfloat16_t;
using ElementLSE = float;
using TileScheduler = kernel::Sm100MlaIndividualTileScheduler;

using MLAKernel = kernel::Sm100FmhaMlaKernelTmaWarpspecialized<TileShape,
                                                               Element,
                                                               ElementAcc,
                                                               ElementOut,
                                                               ElementLSE,
                                                               TileScheduler>;
using MLADevice = device::MLA<MLAKernel>;

/**
 * MLA attention kernel wrapper for testing
 *
 * This wrapper adapts the PyTorch tensor interface to the MLA kernel's
 * argument structure.
 *
 * Input tensors:
 * - q_nope_pe: [batch, num_heads, head_dim_total=576] - Combined Q_latent and
 * Q_rope
 * - ckv_kpe_cache: [num_pages, page_size, head_dim_total=576] - Paged KV cache
 * - kv_len: [batch] - Sequence lengths per batch
 * - page_table: [batch, max_pages] - Page indices for each batch
 * - output: [batch, num_heads, head_dim_latent=512] - Output tensor
 * - workspace: Workspace buffer for split-KV reduction
 * - softmax_scale: Softmax scaling factor
 */
void mla_attention_kernel(
    torch::Tensor q_nope_pe,     // [batch, num_heads, head_dim_total=576]
    torch::Tensor ckv_kpe_cache, // [num_pages, page_size, head_dim_total=576]
    torch::Tensor kv_len,        // [batch]
    torch::Tensor page_table,    // [batch, max_pages]
    torch::Tensor output,        // [batch, num_heads, head_dim_latent=512]
    torch::Tensor workspace,     // workspace buffer
    float softmax_scale) {

  // Get dimensions
  int batch_size = q_nope_pe.size(0);
  int num_heads = q_nope_pe.size(1);
  int head_dim_total = q_nope_pe.size(2);

  int num_pages = ckv_kpe_cache.size(0);
  int page_size = ckv_kpe_cache.size(1);
  int max_pages_per_batch = page_table.size(1);

  TORCH_CHECK(num_heads == NUM_HEADS, "num_heads must be 128 for MLA");
  TORCH_CHECK(head_dim_total == HEAD_DIM_TOTAL,
              "head_dim must be 576 (512 + 64)");
  TORCH_CHECK(q_nope_pe.dtype() == torch::kBFloat16,
              "q_nope_pe must be bfloat16");
  TORCH_CHECK(ckv_kpe_cache.dtype() == torch::kBFloat16,
              "ckv_kpe_cache must be bfloat16");

  // Split q_nope_pe into q_latent and q_rope
  // q_nope_pe layout: [batch, num_heads, 512 + 64]
  auto q_latent = q_nope_pe.slice(2, 0, HEAD_DIM_LATENT).contiguous();
  auto q_rope =
      q_nope_pe.slice(2, HEAD_DIM_LATENT, HEAD_DIM_TOTAL).contiguous();

  // Split ckv_kpe_cache into c_latent and k_rope
  // ckv_kpe_cache layout: [num_pages, page_size, 512 + 64]
  auto c_latent = ckv_kpe_cache.slice(2, 0, HEAD_DIM_LATENT).contiguous();
  auto k_rope =
      ckv_kpe_cache.slice(2, HEAD_DIM_LATENT, HEAD_DIM_TOTAL).contiguous();

  // Compute max sequence length from page table
  int max_seq_len = max_pages_per_batch * page_size;

  // Get hardware info
  KernelHardwareInfo hw_info;
  hw_info.device_id = q_nope_pe.device().index();
  cudaDeviceGetAttribute(
      &hw_info.sm_count, cudaDevAttrMultiProcessorCount, hw_info.device_id);

  // Build kernel arguments
  // ProblemShape = (num_heads=128, seqlen, (d_latent=512, d_rope=64),
  // batch_count)
  typename MLAKernel::ProblemShape problem_shape =
      make_shape(Int<NUM_HEADS>{},
                 max_seq_len,
                 make_shape(Int<HEAD_DIM_LATENT>{}, Int<HEAD_DIM_ROPE>{}),
                 batch_size);

  // TensorStride = Stride<int64_t, _1, int64_t> for (num_heads/seqlen,
  // head_dim, batch) Q tensors: [batch, num_heads, head_dim] -> stride
  // (head_dim, 1, num_heads * head_dim)
  typename MLAKernel::TensorStride stride_q_latent = make_tuple(
      static_cast<int64_t>(HEAD_DIM_LATENT), // stride for num_heads
      _1{},                                  // stride for head_dim (always 1)
      static_cast<int64_t>(num_heads * HEAD_DIM_LATENT) // stride for batch
  );
  typename MLAKernel::TensorStride stride_q_rope =
      make_tuple(static_cast<int64_t>(HEAD_DIM_ROPE),
                 _1{},
                 static_cast<int64_t>(num_heads * HEAD_DIM_ROPE));

  // KV cache tensors: [num_pages, page_size, head_dim] -> stride (head_dim, 1,
  // page_size * head_dim) For paged attention, we treat it as [page_count,
  // page_size, head_dim]
  typename MLAKernel::TensorStride stride_c_latent = make_tuple(
      static_cast<int64_t>(HEAD_DIM_LATENT), // stride for seqlen (within page)
      _1{},                                  // stride for head_dim
      static_cast<int64_t>(page_size * HEAD_DIM_LATENT) // stride for page
  );
  typename MLAKernel::TensorStride stride_k_rope =
      make_tuple(static_cast<int64_t>(HEAD_DIM_ROPE),
                 _1{},
                 static_cast<int64_t>(page_size * HEAD_DIM_ROPE));

  // Output tensor: [batch, num_heads, head_dim_latent]
  typename MLAKernel::TensorStride stride_o =
      make_tuple(static_cast<int64_t>(HEAD_DIM_LATENT),
                 _1{},
                 static_cast<int64_t>(num_heads * HEAD_DIM_LATENT));

  // Build MainloopArguments
  typename MLAKernel::MainloopArguments mainloop_args;
  mainloop_args.softmax_scale = softmax_scale;
  mainloop_args.ptr_q_latent = reinterpret_cast<Element *>(q_latent.data_ptr());
  mainloop_args.stride_q_latent = stride_q_latent;
  mainloop_args.ptr_q_rope = reinterpret_cast<Element *>(q_rope.data_ptr());
  mainloop_args.stride_q_rope = stride_q_rope;
  mainloop_args.ptr_c_latent = reinterpret_cast<Element *>(c_latent.data_ptr());
  mainloop_args.stride_c_latent = stride_c_latent;
  mainloop_args.ptr_k_rope = reinterpret_cast<Element *>(k_rope.data_ptr());
  mainloop_args.stride_k_rope = stride_k_rope;

  // Paged attention setup
  mainloop_args.ptr_seq = kv_len.data_ptr<int>();
  mainloop_args.ptr_page_table = page_table.data_ptr<int>();
  mainloop_args.stride_page_table = make_tuple(_1{}, max_pages_per_batch);
  mainloop_args.page_count = num_pages;
  mainloop_args.page_size = page_size;

  // Build EpilogueArguments
  typename MLAKernel::EpilogueArguments epilogue_args;
  epilogue_args.ptr_o = reinterpret_cast<ElementOut *>(output.data_ptr());
  epilogue_args.stride_o = stride_o;
  epilogue_args.ptr_lse = nullptr; // Not outputting LSE
  epilogue_args.stride_lse = make_tuple(_1{}, num_heads);
  epilogue_args.output_scale = 1.0f;

  // Build full Arguments
  typename MLAKernel::Arguments args;
  args.problem_shape = problem_shape;
  args.mainloop = mainloop_args;
  args.epilogue = epilogue_args;
  args.hw_info = hw_info;
  args.split_kv = 1; // No split-KV for now
  args.ptr_split_kv = nullptr;

  // Check if kernel can implement this configuration
  Status status = MLADevice::can_implement(args);
  if (status != Status::kSuccess) {
    TORCH_CHECK(false, "MLA kernel cannot implement this configuration");
  }

  // Get workspace size and check
  size_t workspace_size = MLADevice::get_workspace_size(args);
  void *workspace_ptr = nullptr;
  if (workspace_size > 0) {
    TORCH_CHECK(workspace.numel() * workspace.element_size() >=
                    static_cast<int64_t>(workspace_size),
                "Workspace too small: need ",
                workspace_size,
                " bytes");
    workspace_ptr = workspace.data_ptr();
  }

  // Create device wrapper and run
  MLADevice mla_device;
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  status = mla_device.run(args, workspace_ptr, stream);

  if (status != Status::kSuccess) {
    TORCH_CHECK(false, "MLA kernel execution failed");
  }

  // Synchronize to catch any kernel errors
  cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    TORCH_CHECK(
        false, "CUDA synchronization failed: ", cudaGetErrorString(err));
  }
}

/**
 * Get workspace size for MLA kernel
 */
int64_t get_workspace_size(int batch_size, int max_seq_len) {
  // Get hardware info
  KernelHardwareInfo hw_info;
  hw_info.device_id = 0;
  cudaDeviceGetAttribute(
      &hw_info.sm_count, cudaDevAttrMultiProcessorCount, hw_info.device_id);

  // Build minimal arguments to query workspace size
  typename MLAKernel::ProblemShape problem_shape =
      make_shape(Int<NUM_HEADS>{},
                 max_seq_len,
                 make_shape(Int<HEAD_DIM_LATENT>{}, Int<HEAD_DIM_ROPE>{}),
                 batch_size);

  typename MLAKernel::Arguments args;
  args.problem_shape = problem_shape;
  args.hw_info = hw_info;
  args.split_kv = 1;

  // Set split_kv automatically
  MLADevice::set_split_kv(args);

  return static_cast<int64_t>(MLADevice::get_workspace_size(args));
}

/**
 * Check if MLA kernel is available on this device
 */
bool is_mla_available() {
  int device;
  cudaGetDevice(&device);

  int major, minor;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);

  // MLA kernel requires SM100 (Blackwell) or later
  return (major >= 10);
}

/**
 * Get MLA kernel configuration info
 */
std::string get_mla_info() {
  std::stringstream ss;
  ss << "MLA Kernel Configuration:\n";
  ss << "  NUM_HEADS: " << NUM_HEADS << "\n";
  ss << "  HEAD_DIM_LATENT: " << HEAD_DIM_LATENT << "\n";
  ss << "  HEAD_DIM_ROPE: " << HEAD_DIM_ROPE << "\n";
  ss << "  HEAD_DIM_TOTAL: " << HEAD_DIM_TOTAL << "\n";
  ss << "  TILE_S: " << TILE_S << "\n";
  ss << "  Element type: bfloat16\n";
  ss << "  Accumulator type: float\n";
  ss << "  SharedStorageSize: " << MLAKernel::SharedStorageSize << " bytes\n";
  return ss.str();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mla_attention",
        &mla_attention_kernel,
        "MLA (Multi-head Latent Attention) kernel for Blackwell (SM100)",
        py::arg("q_nope_pe"),
        py::arg("ckv_kpe_cache"),
        py::arg("kv_len"),
        py::arg("page_table"),
        py::arg("output"),
        py::arg("workspace"),
        py::arg("softmax_scale"));

  m.def("get_workspace_size",
        &get_workspace_size,
        "Get workspace size for MLA kernel",
        py::arg("batch_size"),
        py::arg("max_seq_len"));

  m.def("is_mla_available",
        &is_mla_available,
        "Check if MLA kernel is available on this device");

  m.def("get_mla_info", &get_mla_info, "Get MLA kernel configuration info");
}
