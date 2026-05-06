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

// DeepSeek V4 C4 KV Compressor / Cache Insert for SM100.
//
// This is intentionally a compile-safe skeleton. It wires the new DeepSeek V4
// C4 compressor task into MPK, documents the exact contract, and leaves the
// CUDA math as TODOs for the follow-up implementation PR.
//
// DeepSeek V4 Flash Base constants targeted by this first task:
//   compress_ratio = 4
//   overlap        = true
//   head_dim       = 512
//   rope_head_dim  = 64
//   nope_head_dim  = 448
//   coff           = 2
//   kv_score_dim   = 4 * head_dim = 2048
//
// Input pointer contract:
//   input_ptrs[0] kv_score:
//     [max_num_batched_tokens, KV_SCORE_DIM]
//     TODO dtype decision: official HF computes compression in fp32. MPK may
//     feed fp32 or bf16 depending on the eventual fused wkv/wgate path.
//
//   input_ptrs[1] token_meta:
//     int32 [max_num_batched_tokens, 2]
//     token_meta[token, 0] = absolute sequence position
//     token_meta[token, 1] = physical C4 cache slot, or -1 if this token does
//                            not emit a compressed KV entry.
//
//   input_ptrs[2] state_cache:
//     float32 [max_requests, 8, KV_SCORE_DIM]
//     Stores C4 overlap/current state. The final implementation must keep this
//     layout compatible with vLLM CompressorStateCache semantics, while fitting
//     MPK's persistent-kernel metadata model.
//
//   input_ptrs[3] c4_cache:
//     bf16 [num_c4_pages, C4_PAGE_SIZE, HEAD_DIM]
//     Correctness-first cache format. A later performance pass should switch
//     to FlashMLA-compatible FP8-with-scale cache:
//       512 fp8 NoPE bytes + 4 fp32 scales + 64 bf16 RoPE values.
//
//   input_ptrs[4] ape:
//     float32 [8, HEAD_DIM]
//     Prepacked from the official HF ape [4, 1024]:
//       rows 0..3 = overlap half, rows 4..7 = current half.
//
//   input_ptrs[5] norm_weight:
//     bf16 or fp32 [HEAD_DIM]
//
//   input_ptrs[6] rope_cos_sin:
//     float32 [max_seq_len, ROPE_HEAD_DIM], GPT-J/interleaved style layout
//     with first half cos and second half sin.
//
// Runtime metadata:
//   qo_indptr_buffer[request_id:request_id+2] selects this request's token
//   window in kv_score/token_meta, matching existing MPK MLA gather tasks.
//
// DeepSeek official semantic TODOs, from HF inference/model.py Compressor:
//   1. Prefill: cutoff = seqlen - (seqlen % 4); remainder tokens stay in state.
//   2. C4 overlap transform: each compressed block uses 8 slots:
//      previous block's 4 overlap slots + current block's 4 current slots.
//   3. The first block has invalid overlap; use KV = 0 and score = -inf.
//   4. Decode: should_compress = ((absolute_position + 1) % 4 == 0).
//   5. Add APE before softmax, using absolute_position % 4 for decode.
//   6. Compute stable softmax over the 8 scores per hidden dimension and form
//      the weighted KV sum.
//   7. Apply RMSNorm after weighted pooling.
//   8. Apply RoPE only to the last ROPE_HEAD_DIM elements at position
//      absolute_position + 1 - 4.
//   9. Write compressed KV to c4_cache[token_meta[token, 1]].
//  10. After a write, shift current state into overlap state for the next C4.
//
// Implementation references:
//   - DeepSeek HF inference/model.py: Compressor.forward exact math.
//   - SGLang deepseek_v4 c4.cuh: 8-slot window and online softmax shape.
//   - vLLM deepseek_compressor.py: state cache and fused insert metadata.
//   - FlashMLA README: future physical index and FP8 KV cache format.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace kernel {

template <int HEAD_DIM, int ROPE_HEAD_DIM, int KV_SCORE_DIM, int C4_PAGE_SIZE>
__device__ __forceinline__ void dsv4_c4_compress_sm100_task_impl(
    void const *kv_score_ptr,
    void const *token_meta_ptr,
    void *state_cache_ptr,
    void *c4_cache_ptr,
    void const *ape_ptr,
    void const *norm_weight_ptr,
    void const *rope_cos_sin_ptr,
    int const *qo_indptr_buffer_ptr,
    int request_id) {
  static_assert(HEAD_DIM == 512,
                "DeepSeek V4 Flash Base C4 skeleton only supports head_dim=512");
  static_assert(ROPE_HEAD_DIM == 64,
                "DeepSeek V4 Flash Base C4 skeleton only supports rope_dim=64");
  static_assert(KV_SCORE_DIM == 4 * HEAD_DIM,
                "C4 kv_score layout must be [kv_overlap, kv, score_overlap, score]");
  static_assert(C4_PAGE_SIZE > 0, "C4 cache page size must be positive");

  // TODO(dpskv4): Implement the C4 compressor/cache-insert body.
  //
  // This stub deliberately does not touch state_cache or c4_cache yet. It keeps
  // the task registration, code generation, and CUDA compilation path alive so
  // the follow-up kernel implementation can focus on correctness and perf.
  (void)kv_score_ptr;
  (void)token_meta_ptr;
  (void)state_cache_ptr;
  (void)c4_cache_ptr;
  (void)ape_ptr;
  (void)norm_weight_ptr;
  (void)rope_cos_sin_ptr;
  (void)qo_indptr_buffer_ptr;
  (void)request_id;
}

} // namespace kernel
