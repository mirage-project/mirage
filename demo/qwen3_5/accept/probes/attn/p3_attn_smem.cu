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

// Probe P3 (v1-architecture.md §14) — attention smem instantiation sweep.
//
// A bare translation unit that instantiates the SM100 paged-attention task impl
// at the Qwen3.5-35B-A3B full-attention shape and lets the kernel's own
// `static_assert(S_TOTAL_OFFSET <= MAX_DYNAMIC_SHARED_MEMORY_SIZE)` decide
// whether MAX_TOKENS is admissible. This is the same validation method commit
// 5715c6f used ("nvcc -arch=sm_100a compile check instantiating the kernel with
// Qwen3-32B GQA 8:1 parameters: fails the static_assert before this change,
// compiles cleanly after").
//
// The instantiation must be *odr-used* from a __global__ entry point: the impl
// is `__device__ __forceinline__` and templated, so a mere declaration would
// never instantiate the body and the static_assert would never fire.
//
// Shape (docs/qwen35/vllm-graph.md §2.2.1, v1-architecture.md §2.3):
//   16 Q heads / 2 KV heads  -> NUM_QO_PER_KV = 8   (GQA 8:1)
//   head_dim                 = 256
//   KV cache is (pages, page_size, 2, 256) NHD    -> KV_CACHE_STRIDE = 512
//   o                        = [T, 16*256 = 4096] -> O_STRIDE        = 4096
//
// QKV_STRIDE covers both attention variants of this issue:
//   * base (no output gate): 2 * (8*256 + 256 + 256) = 5120
//   * QKVG (per-head [q|gate], §4.2): 2 * (8*512 + 256 + 256) = 9216
// It does not enter any smem size, so the sweep result is identical for both;
// -DP3_QKV_STRIDE overrides it for an explicit cross-check.
//
// Build (one TU per MAX_TOKENS value):
//   nvcc -arch=sm_100a -std=c++17 --expt-relaxed-constexpr \
//        -DMPK_TARGET_CC=100 -DMIRAGE_GRACE_BLACKWELL -DMODE_OFFLINE \
//        -DMIRAGE_BACKEND_USE_CUDA -DP3_MAX_TOKENS=<MT> \
//        -I include -I include/mirage/persistent_kernel \
//        -I include/mirage/persistent_kernel/tasks \
//        -I deps/cutlass/include -I deps/cutlass/tools/util/include \
//        -c p3_attn_smem.cu -o /dev/null
//
// Exit status 0 => COMPILES (MAX_TOKENS admissible); non-zero with the
// static_assert diagnostic => STATIC_ASSERT (over the 201 KiB budget).

#include "blackwell/attention_sm100.cuh"

#ifndef P3_MAX_TOKENS
#define P3_MAX_TOKENS 4
#endif

#ifndef P3_NUM_QO_PER_KV
#define P3_NUM_QO_PER_KV 8
#endif

#ifndef P3_HEAD_DIM
#define P3_HEAD_DIM 256
#endif

// 2 KV heads * 256 head_dim (packed K|V cache view, vllm-graph.md §4.2).
#ifndef P3_KV_CACHE_STRIDE
#define P3_KV_CACHE_STRIDE 512
#endif

// QKVG-fused row width (§4.2). Not used in any smem term.
#ifndef P3_QKV_STRIDE
#define P3_QKV_STRIDE 9216
#endif

#ifndef P3_O_STRIDE
#define P3_O_STRIDE 4096
#endif

#ifndef P3_MAX_SEQ_LEN
#define P3_MAX_SEQ_LEN 2048
#endif

#ifndef P3_PAGE_SIZE
#define P3_PAGE_SIZE 64
#endif

using bfloat16 = type::bfloat16_t;

// Report the derived smem arena so the probe artifact carries measured numbers
// rather than a re-derivation of the paper model. Every term mirrors
// attention_sm100.cuh's constexpr block exactly.
namespace p3 {
constexpr int MAX_TOKENS = P3_MAX_TOKENS;
constexpr int NUM_QO_PER_KV = P3_NUM_QO_PER_KV;
constexpr int HEAD_DIM = P3_HEAD_DIM;
constexpr int KV_TILE_SIZE = 64;
constexpr int MMA_ITERS_M = (MAX_TOKENS * NUM_QO_PER_KV + 15) / 16;

constexpr size_t ZERO_BUFFER_SIZE = sizeof(bfloat16) * 8;
constexpr size_t S_Q_SIZE =
    sizeof(bfloat16) * MAX_TOKENS * NUM_QO_PER_KV * HEAD_DIM;
constexpr size_t S_K_SIZE = sizeof(bfloat16) * KV_TILE_SIZE * HEAD_DIM;
constexpr size_t S_O_SIZE = S_Q_SIZE;
constexpr size_t S_NORM_SUM_SIZE = sizeof(float) * 4 * 2;
constexpr size_t S_MD_BUFFER_SIZE =
    sizeof(float) * MMA_ITERS_M * NUM_THREADS * 2 * 2;
// Post-5715c6f: one m-tile at a time. Pre-pick this was * MMA_ITERS_M.
constexpr size_t S_O_BUFFER_SIZE_POST_PICK = sizeof(float) * NUM_THREADS * 64;
constexpr size_t S_O_BUFFER_SIZE_PRE_PICK =
    sizeof(float) * MMA_ITERS_M * NUM_THREADS * 64;

constexpr size_t TOTAL_POST_PICK = ZERO_BUFFER_SIZE + S_Q_SIZE + 4 * S_K_SIZE +
                                   S_O_SIZE + S_NORM_SUM_SIZE +
                                   S_MD_BUFFER_SIZE + S_O_BUFFER_SIZE_POST_PICK;
constexpr size_t TOTAL_PRE_PICK = ZERO_BUFFER_SIZE + S_Q_SIZE + 4 * S_K_SIZE +
                                  S_O_SIZE + S_NORM_SUM_SIZE +
                                  S_MD_BUFFER_SIZE + S_O_BUFFER_SIZE_PRE_PICK;
} // namespace p3

#ifdef P3_EMIT_SIZES
// Extract the arena sizes FROM THE COMPILER rather than re-deriving them in the
// runner: instantiating an incomplete class template with the size as a
// non-type argument makes nvcc name the value in its diagnostic, e.g.
//   error: ... incomplete type "P3_ARENA<174128UL>"
// The runner greps those integers out. The p3:: copies above are only a mirror
// of attention_sm100.cuh's constexpr block, so the runner ALSO cross-checks
// them against the ground truth (the observed COMPILES/STATIC_ASSERT boundary
// of the real kernel) instead of trusting the mirror.
template <size_t N>
struct P3_ARENA;
P3_ARENA<p3::TOTAL_POST_PICK> p3_emit_post_pick;
P3_ARENA<p3::TOTAL_PRE_PICK> p3_emit_pre_pick;
P3_ARENA<(size_t)mirage::runtime::MAX_DYNAMIC_SHARED_MEMORY_SIZE>
    p3_emit_budget;
P3_ARENA<p3::MMA_ITERS_M> p3_emit_mma_iters_m;
#endif

__global__ void __launch_bounds__(WORKER_NUM_THREADS)
    p3_attn_smem_probe(void const *qkv_ptr,
                       void *paged_k_cache_ptr,
                       void *paged_v_cache_ptr,
                       void *output_ptr,
                       int const *qo_indptr_buffer_ptr,
                       int const *paged_kv_indptr_buffer_ptr,
                       int const *paged_kv_indices_buffer_ptr,
                       int const *paged_kv_last_page_len_buffer_ptr,
                       int16_t request_id,
                       void const *q_norm_weight_ptr,
                       void const *k_norm_weight_ptr,
                       void const *cos_ptr,
                       void const *sin_ptr) {
  kernel::multitoken_paged_attention_sm100_task_impl<bfloat16,
                                                     P3_NUM_QO_PER_KV,
                                                     1,
                                                     P3_KV_CACHE_STRIDE,
                                                     P3_QKV_STRIDE,
                                                     P3_O_STRIDE,
                                                     P3_HEAD_DIM,
                                                     P3_MAX_SEQ_LEN,
                                                     P3_PAGE_SIZE,
                                                     0 /* Q_LEN_OVERRIDE */,
                                                     0 /* TAIL_OFFSET  */,
                                                     P3_MAX_TOKENS>(
      qkv_ptr,
      paged_k_cache_ptr,
      paged_v_cache_ptr,
      output_ptr,
      qo_indptr_buffer_ptr,
      paged_kv_indptr_buffer_ptr,
      paged_kv_indices_buffer_ptr,
      paged_kv_last_page_len_buffer_ptr,
      request_id,
      true /* qk_norm */,
      true /* rope    */,
      q_norm_weight_ptr,
      k_norm_weight_ptr,
      cos_ptr,
      sin_ptr,
      1e-6f,
      1e-6f);
}
