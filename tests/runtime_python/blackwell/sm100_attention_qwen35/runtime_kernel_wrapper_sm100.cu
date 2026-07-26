/* Copyright 2026 CMU
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

// Kernel-wrapper test harness for the SM100 paged-attention task's Qwen3.5
// variants
// (include/mirage/persistent_kernel/tasks/blackwell/attention_sm100.cuh):
//   * ATTN_OUTPUT_GATE -- fused QKVG input row + `out *= sigmoid(gate)`
//   epilogue
//   * Q_PASS_SIZE      -- the in-kernel Q-loop that decouples the smem arena
//                         from max-batched-tokens
//
// The launch mirrors what `TaskRegister::register_paged_attention_sm100_task`
// plus the TBGraph partition `(-1, 1, -1)` produce at runtime: grid
// (max_batched_requests, num_kv_heads), one CTA per (request slot, kv group),
// with each pointer pre-offset to that group's slice --
//
//   qkv     += kv_group * QKV_GROUP_WIDTH   (the dim-1 partition of the packed
//   row) k/v cache += kv_group * HEAD_DIM        ((pages, page_size, kv_heads,
//   head_dim)) output  += kv_group * NUM_QO_PER_KV * HEAD_DIM request_id =
//   blockIdx.x
//
// so the wrapper exercises exactly the addressing the generated
// `_execute_task()` uses, not a simplified stand-in.

#include "blackwell/attention_sm100.cuh"
#include "runtime_header.h"
#include <cuda_runtime.h>
#include <torch/extension.h>

using bfloat16 = type::bfloat16_t;

namespace {

template <int NUM_QO_PER_KV,
          int NUM_KV_HEADS,
          int HEAD_DIM,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          int MAX_TOKENS,
          int ATTN_OUTPUT_GATE,
          int Q_PASS_SIZE>
__global__ void __launch_bounds__(WORKER_NUM_THREADS)
    attention_qwen35_wrapper(void const *qkv_ptr,
                             void *k_cache_ptr,
                             void *v_cache_ptr,
                             void *output_ptr,
                             int const *qo_indptr,
                             int const *kv_indptr,
                             int const *kv_indices,
                             int const *kv_last_page_len,
                             void const *q_norm_weight_ptr,
                             void const *k_norm_weight_ptr,
                             void const *cos_ptr,
                             void const *sin_ptr,
                             bool qk_norm,
                             bool rope) {
  // Per-kv-group widths, exactly as the packed row is laid out.
  constexpr int Q_HEAD_STRIDE = ATTN_OUTPUT_GATE ? (2 * HEAD_DIM) : HEAD_DIM;
  constexpr int QKV_GROUP_WIDTH = NUM_QO_PER_KV * Q_HEAD_STRIDE + 2 * HEAD_DIM;
  constexpr int QKV_STRIDE = QKV_GROUP_WIDTH * NUM_KV_HEADS;
  constexpr int KV_CACHE_STRIDE = NUM_KV_HEADS * HEAD_DIM;
  constexpr int O_STRIDE = NUM_QO_PER_KV * NUM_KV_HEADS * HEAD_DIM;

  int const slot = blockIdx.x;
  int const kv_group = blockIdx.y;

  bfloat16 const *qkv =
      static_cast<bfloat16 const *>(qkv_ptr) + kv_group * QKV_GROUP_WIDTH;
  bfloat16 *kc = static_cast<bfloat16 *>(k_cache_ptr) + kv_group * HEAD_DIM;
  bfloat16 *vc = static_cast<bfloat16 *>(v_cache_ptr) + kv_group * HEAD_DIM;
  bfloat16 *out =
      static_cast<bfloat16 *>(output_ptr) + kv_group * NUM_QO_PER_KV * HEAD_DIM;

  kernel::multitoken_paged_attention_sm100_task_impl<bfloat16,
                                                     NUM_QO_PER_KV,
                                                     1,
                                                     KV_CACHE_STRIDE,
                                                     QKV_STRIDE,
                                                     O_STRIDE,
                                                     HEAD_DIM,
                                                     MAX_SEQ_LEN,
                                                     PAGE_SIZE,
                                                     0 /* Q_LEN_OVERRIDE */,
                                                     0 /* TAIL_OFFSET */,
                                                     MAX_TOKENS,
                                                     ATTN_OUTPUT_GATE,
                                                     Q_PASS_SIZE>(
      qkv,
      kc,
      vc,
      out,
      qo_indptr,
      kv_indptr,
      kv_indices,
      kv_last_page_len,
      static_cast<int16_t>(slot),
      qk_norm,
      rope,
      q_norm_weight_ptr,
      k_norm_weight_ptr,
      cos_ptr,
      sin_ptr,
      1e-6f,
      1e-6f);
}

template <int NUM_QO_PER_KV,
          int NUM_KV_HEADS,
          int HEAD_DIM,
          int MAX_SEQ_LEN,
          int PAGE_SIZE,
          int MAX_TOKENS,
          int ATTN_OUTPUT_GATE,
          int Q_PASS_SIZE>
void launch(torch::Tensor qkv,
            torch::Tensor k_cache,
            torch::Tensor v_cache,
            torch::Tensor output,
            torch::Tensor qo_indptr,
            torch::Tensor kv_indptr,
            torch::Tensor kv_indices,
            torch::Tensor kv_last_page_len,
            torch::Tensor q_norm_weight,
            torch::Tensor k_norm_weight,
            torch::Tensor cos,
            torch::Tensor sin,
            int num_requests,
            bool qk_norm,
            bool rope) {
  auto fn = attention_qwen35_wrapper<NUM_QO_PER_KV,
                                     NUM_KV_HEADS,
                                     HEAD_DIM,
                                     MAX_SEQ_LEN,
                                     PAGE_SIZE,
                                     MAX_TOKENS,
                                     ATTN_OUTPUT_GATE,
                                     Q_PASS_SIZE>;
  // The task sizes its own arena from MAX_TOKENS; hand it the runtime budget
  // the persistent kernel would (runtime_header.h).
  size_t smem = mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE;
  cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  dim3 grid(num_requests, NUM_KV_HEADS, 1);
  dim3 block(WORKER_NUM_THREADS, 1, 1);
  fn<<<grid, block, smem>>>(qkv.data_ptr(),
                            k_cache.data_ptr(),
                            v_cache.data_ptr(),
                            output.data_ptr(),
                            qo_indptr.data_ptr<int>(),
                            kv_indptr.data_ptr<int>(),
                            kv_indices.data_ptr<int>(),
                            kv_last_page_len.data_ptr<int>(),
                            q_norm_weight.data_ptr(),
                            k_norm_weight.data_ptr(),
                            cos.data_ptr(),
                            sin.data_ptr(),
                            qk_norm,
                            rope);
  cudaError_t err = cudaDeviceSynchronize();
  TORCH_CHECK(err == cudaSuccess,
              "attention_qwen35 launch failed: ",
              cudaGetErrorString(err));
}

} // namespace

// Explicit dispatch. Template arguments cannot come from Python, so every
// tested configuration is enumerated here; an unknown one is a loud error
// rather than a silent fallback to a different shape.
void attention_qwen35(torch::Tensor qkv,
                      torch::Tensor k_cache,
                      torch::Tensor v_cache,
                      torch::Tensor output,
                      torch::Tensor qo_indptr,
                      torch::Tensor kv_indptr,
                      torch::Tensor kv_indices,
                      torch::Tensor kv_last_page_len,
                      torch::Tensor q_norm_weight,
                      torch::Tensor k_norm_weight,
                      torch::Tensor cos,
                      torch::Tensor sin,
                      int num_requests,
                      int num_qo_per_kv,
                      int num_kv_heads,
                      int head_dim,
                      int max_tokens,
                      int attn_output_gate,
                      int q_pass_size,
                      bool qk_norm,
                      bool rope) {
#define CASE(QO, KVH, HD, MT, GATE, QP)                                        \
  if (num_qo_per_kv == QO && num_kv_heads == KVH && head_dim == HD &&          \
      max_tokens == MT && attn_output_gate == GATE && q_pass_size == QP) {     \
    launch<QO, KVH, HD, 2048, 64, MT, GATE, QP>(qkv,                           \
                                                k_cache,                       \
                                                v_cache,                       \
                                                output,                        \
                                                qo_indptr,                     \
                                                kv_indptr,                     \
                                                kv_indices,                    \
                                                kv_last_page_len,              \
                                                q_norm_weight,                 \
                                                k_norm_weight,                 \
                                                cos,                           \
                                                sin,                           \
                                                num_requests,                  \
                                                qk_norm,                       \
                                                rope);                         \
    return;                                                                    \
  }

  // ---- Qwen3.5-35B-A3B full attention: 16 Q / 2 KV heads, head_dim 256 ----
  // MAX_TOKENS = 4 is the largest value that fits the 201 KiB budget at this
  // shape post-5715c6f (probe P3).
  CASE(8, 2, 256, 4, 1, 0) // gated, single pass (decode)
  CASE(8, 2, 256, 4, 1, 4) // gated + Q-loop (prefill chunk)
  CASE(8, 2, 256, 4, 0, 0) // UNGATED control -> isolates core_attn_out
  CASE(8, 2, 256, 4, 0, 4) // ungated + Q-loop
  CASE(8, 2, 256, 2, 1, 2) // gated + finer pass split (pass-size invariance)
  CASE(8, 2, 256, 1, 1, 1) // gated, one query per pass

  // ---- small shape used for the Q-loop equivalence proof --------------
  // 4 Q / 1 KV head, head_dim 128: MAX_TOKENS=8 fits here, so a single 8-row
  // pass can be compared bit-for-bit against 2x4 and 4x2 passes.
  CASE(4, 1, 128, 8, 0, 0) // reference: one pass of 8
  CASE(4, 1, 128, 8, 0, 4) // same arena, two passes of 4
  CASE(4, 1, 128, 4, 0, 4) // production form: arena 4, two passes of 4
  CASE(4, 1, 128, 2, 0, 2) // arena 2, four passes of 2
  CASE(4, 1, 128, 8, 1, 0) // gated variants of the same pair
  CASE(4, 1, 128, 4, 1, 4)
#undef CASE
  TORCH_CHECK(false,
              "unsupported attention_qwen35 config: num_qo_per_kv=",
              num_qo_per_kv,
              " num_kv_heads=",
              num_kv_heads,
              " head_dim=",
              head_dim,
              " max_tokens=",
              max_tokens,
              " gate=",
              attn_output_gate,
              " q_pass_size=",
              q_pass_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("attention_qwen35",
        &attention_qwen35,
        "SM100 paged attention -- Qwen3.5 QKVG-gate / Q-loop variants");
}
