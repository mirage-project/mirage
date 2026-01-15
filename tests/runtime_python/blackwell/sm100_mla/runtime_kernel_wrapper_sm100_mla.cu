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

#include <cuda_runtime.h>
#include <torch/extension.h>
#include <iostream>

#include "blackwell/attention_mla_sm100.cuh"

using bfloat16 = cutlass::bfloat16_t;

// MLA attention kernel wrapper for testing
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

  constexpr int head_dim_latent = 512;
  constexpr int head_dim_rope = 64;

  TORCH_CHECK(num_heads == 128, "num_heads must be 128 for MLA");
  TORCH_CHECK(head_dim_total == head_dim_latent + head_dim_rope,
              "head_dim must be 576 (512 + 64)");

  // Split q_nope_pe into q_latent and q_rope
  // q_nope_pe layout: [batch, num_heads, 512 + 64]
  auto q_latent = q_nope_pe.slice(2, 0, head_dim_latent).contiguous();
  auto q_rope = q_nope_pe.slice(2, head_dim_latent, head_dim_total).contiguous();

  // Split ckv_kpe_cache into c_latent and k_rope
  // ckv_kpe_cache layout: [num_pages, page_size, 512 + 64]
  auto c_latent = ckv_kpe_cache.slice(2, 0, head_dim_latent).contiguous();
  auto k_rope = ckv_kpe_cache.slice(2, head_dim_latent, head_dim_total).contiguous();

  // Compute max sequence length from page table
  int max_seq_len = max_pages_per_batch * page_size;

  // Build MLA arguments
  mirage::mla::MLAArguments<bfloat16> args;
  args.ptr_q_latent = reinterpret_cast<bfloat16*>(q_latent.data_ptr());
  args.ptr_q_rope = reinterpret_cast<bfloat16*>(q_rope.data_ptr());
  args.ptr_c_latent = reinterpret_cast<bfloat16*>(c_latent.data_ptr());
  args.ptr_k_rope = reinterpret_cast<bfloat16*>(k_rope.data_ptr());
  args.ptr_output = reinterpret_cast<bfloat16*>(output.data_ptr());
  args.ptr_lse = nullptr;  // LSE not needed for basic test
  args.ptr_seq_lens = kv_len.data_ptr<int>();
  args.ptr_page_table = page_table.data_ptr<int>();
  args.page_size = page_size;
  args.page_count = num_pages;
  args.batch_size = batch_size;
  args.max_seq_len = max_seq_len;
  args.softmax_scale = softmax_scale;
  args.split_kv = -1;  // Auto
  args.workspace = workspace.data_ptr();
  args.workspace_size = workspace.numel() * workspace.element_size();

  // Launch MLA kernel
  cudaError_t err = mirage::mla::launch_mla_attention<>(args);
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "MLA kernel launch failed: ", cudaGetErrorString(err));
  }

  // Synchronize
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    TORCH_CHECK(false, "CUDA synchronization failed: ", cudaGetErrorString(err));
  }
}

// Get required workspace size
int64_t get_mla_workspace_size(int batch_size, int max_seq_len) {
  return static_cast<int64_t>(
      mirage::mla::get_mla_workspace_size<>(batch_size, max_seq_len));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mla_attention", &mla_attention_kernel,
        "MLA attention kernel for DeepSeek-style Multi-head Latent Attention");
  m.def("get_workspace_size", &get_mla_workspace_size,
        "Get required workspace size for MLA attention");
}
