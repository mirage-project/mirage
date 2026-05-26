/* Copyright 2025 Mirage Team
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
#include "../common/utils.cuh"
#include "../common/worker_config.h"
namespace kernel {

// ============================================================================
// moe_permute_sm100
// ============================================================================
//
// Build the pre-permuted input layout that the PR-674 grouped FP8 GEMM
// (`fp8_group_gemm_smallm/largem_sm100`) consumes, while keeping the
// MPK-level builder interface aligned with the OLD MoE path
// (`routing_indices`, `mask`, `(mbt, K)` input, `(mbt, hidden)` output).
//
// One CTA per local expert (grid_dim = (E_LOCAL, 1, 1)). The expert id is
// read from `task_desc->task_metadata.expert_offset = bid.x` (set by the
// runtime — see runtime.cc near the other MoE-task `expert_offset`
// assignments).
//
// MPK output-port limit is 3 per task, so this task emits the bulky FP8
// data + scale as separate outputs and packs the small per-row metadata
// (weights + token-id) into ONE shared `meta` buffer:
//
//   meta layout (int32 buffer, total length M_TOTAL + MBT*TOPK):
//     meta[0          : M_TOTAL]              = permuted_weights (float32
//                                                reinterpret-cast as int32)
//     meta[M_TOTAL    : M_TOTAL + MBT*TOPK]   = token_to_permuted (int32,
//                                                row + 1; 0 = not routed
//                                                locally, set by upstream
//                                                tensor_init zero-fill)
//
// `m_indices` is NOT emitted by this task — it's a static constant
// (m_indices[r] = r / BM_PADDING) that the builder allocates once at task
// graph build time via attach_input and feeds straight to the grouped
// GEMM. Saves one output port.
//
// Buffer layout (compile-time M_TOTAL = E_LOCAL * BM_PADDING):
//   input_fp8           : (MBT, K)         uint8
//   input_scale         : (MBT, K_PACKED)  uint32           — UE8M0 packed
//   topk_weights        : (MBT, TOPK)      float32
//   routing_indices     : (E_LOCAL, MBT)   int32            — topk_sigmoid
//   output permuted_fp8 (out)  : (M_TOTAL, K)     uint8 permuted_scale (out):
//   (K_PACKED, M_TOTAL) uint32        — TRANSPOSED packed meta (out)          :
//   (M_TOTAL + MBT*TOPK,) int32       — see layout above
//

template <int K, int K_PACKED, int MBT, int TOPK, int E_LOCAL, int BM_PADDING>
__device__ __forceinline__ void
    moe_permute_sm100_task_impl(void const *input_fp8_ptr,
                                void const *input_scale_ptr,
                                void const *topk_weights_ptr,
                                void const *routing_indices_ptr,
                                void *permuted_fp8_ptr,
                                void *permuted_scale_ptr,
                                void *meta_ptr,
                                int my_expert,
                                int num_active_rows) {
  uint8_t const *__restrict__ in_fp8 =
      static_cast<uint8_t const *>(input_fp8_ptr);
  uint32_t const *__restrict__ in_scale =
      static_cast<uint32_t const *>(input_scale_ptr);
  float const *__restrict__ in_weights =
      static_cast<float const *>(topk_weights_ptr);
  int32_t const *__restrict__ routing =
      static_cast<int32_t const *>(routing_indices_ptr);

  uint8_t *__restrict__ out_fp8 = static_cast<uint8_t *>(permuted_fp8_ptr);
  uint32_t *__restrict__ out_scale =
      static_cast<uint32_t *>(permuted_scale_ptr);
  int32_t *__restrict__ meta = static_cast<int32_t *>(meta_ptr);

  // Sub-regions inside the meta buffer.
  // The meta tensor is shape (2, M_TOTAL + MBT*TOPK) int32 — see the wrapper
  // docstring. Row 0 holds out_weights (float32 bits) + tok_to_perm; row 1
  // was unused historically (tensor_init artifact) and now carries the
  // per-expert active mask consumed by fp8_group_gemm to skip whole-expert
  // tile blocks where no token routed locally.
  constexpr int M_TOTAL = E_LOCAL * BM_PADDING;
  constexpr int META_ROW_STRIDE = M_TOTAL + MBT * TOPK;
  float *__restrict__ out_weights =
      reinterpret_cast<float *>(meta);                // [0 : M_TOTAL)
  int32_t *__restrict__ tok_to_perm = meta + M_TOTAL; // [M_TOTAL : ...)
  int32_t *__restrict__ active_expert_mask =
      meta + META_ROW_STRIDE; // [META_ROW_STRIDE : META_ROW_STRIDE + E_LOCAL)

  int const my_row_base = my_expert * BM_PADDING;
  int const tid = threadIdx.x;
  int const nthreads = blockDim.x;

  // Phase 1: scan routing_indices[my_expert, 0..MBT-1] deterministically.
  // C16 (2026-05-17): warp-parallel via __ballot_sync + __popc, still
  // deterministic. Each iter of the for loop processes 32 consecutive
  // positions; lane 0 atomicAdd's the chunk count into s_count, which
  // preserves chunk order (warp executes iterations sequentially). Within
  // a chunk, lane_idx == bit-position in __ballot_sync mask, so
  // popc(mask & ((1<<lane)-1)) gives the deterministic intra-chunk slot.
  __shared__ int s_count;
  __shared__ int s_matched_token[BM_PADDING];
  __shared__ int s_matched_slot[BM_PADDING]; // 0-indexed (routing_val - 1)

  if (tid == 0) {
    s_count = 0;
  }
  __syncthreads();

  // Only warp 0 cooperates on the scan. Other warps idle here — Phase 2
  // re-engages them. This trades 13 μs of single-thread work for ~1 μs
  // of warp-parallel work (scan_end=128 → 4 chunks of 32 lanes).
  {
    int const my_routing_base = my_expert * MBT;
    int const scan_end = (num_active_rows < MBT) ? num_active_rows : MBT;
    if (tid < 32) {
      int const lane = tid;
#pragma unroll 1
      for (int chunk_base = 0; chunk_base < scan_end; chunk_base += 32) {
        int const t = chunk_base + lane;
        int32_t slot_1idx = (t < scan_end) ? routing[my_routing_base + t] : 0;
        unsigned const mask = __ballot_sync(0xffffffff, slot_1idx > 0);
        int const my_offset = __popc(mask & ((1u << lane) - 1));
        int const chunk_count = __popc(mask);
        int base_slot;
        if (lane == 0) {
          base_slot = s_count;
          s_count += chunk_count;
        }
        base_slot = __shfl_sync(0xffffffff, base_slot, 0);
        int const my_slot = base_slot + my_offset;
        if (slot_1idx > 0 && my_slot < BM_PADDING) {
          s_matched_token[my_slot] = t;
          s_matched_slot[my_slot] = slot_1idx - 1;
        }
      }
      if (lane == 0 && s_count > BM_PADDING) {
#if MPK_DEBUG
        printf("MOE_PERMUTE overflow: expert s_count=%d cap=%d "
               "(routings beyond cap would be silently dropped)\n",
               s_count,
               BM_PADDING);
        __trap();
#endif
        s_count = BM_PADDING;
      }
    }
  }
  __syncthreads();
  int const actual_count = s_count;

  // Phase 2: per-row copy. Real rows do FP8 + scale + metadata; padded
  // rows get permuted_weights = 0 (so the downstream unpermute drops
  // whatever junk the GEMM wrote there).
  constexpr int CP_BYTES = 16;
  constexpr int NUM_VEC = K / CP_BYTES;
  static_assert(K % CP_BYTES == 0, "K must be 16-byte aligned");

  for (int slot = 0; slot < BM_PADDING; ++slot) {
    int row = my_row_base + slot;
    if (slot < actual_count) {
      int t = s_matched_token[slot];
      int k_slot = s_matched_slot[slot];

      // Copy FP8 row using 16-byte vectorized loads.
      uint4 const *src_v =
          reinterpret_cast<uint4 const *>(in_fp8 + (size_t)t * K);
      uint4 *dst_v = reinterpret_cast<uint4 *>(out_fp8 + (size_t)row * K);
      for (int i = tid; i < NUM_VEC; i += nthreads) {
        dst_v[i] = src_v[i];
      }

      // Transpose-pack the scale row.
      uint32_t const *src_scale = in_scale + (size_t)t * K_PACKED;
      for (int sf = tid; sf < K_PACKED; sf += nthreads) {
        out_scale[(size_t)sf * M_TOTAL + row] = src_scale[sf];
      }

      if (tid == 0) {
        out_weights[row] = in_weights[(size_t)t * TOPK + k_slot];
        tok_to_perm[(size_t)t * TOPK + k_slot] = row + 1; // 1-indexed
      }
    } else if (tid == 0) {
      // Padding row: weight = 0 so unpermute ignores; junk FP8 OK because
      // the GEMM output for these rows will be multiplied by 0.
      out_weights[row] = 0.0f;
    }
  }

  // Phase 3: publish per-expert active mask + actual row count.
  //
  // active_expert_mask[expert]: 0/1 — used by fp8_group_gemm + silu_mul
  // D1/D3 short-circuits.
  //
  // actual_count_per_expert[expert]: number of real (non-padding) routed
  // rows this iter. Decode (active_token=1) routes at most 1 row per
  // selected expert; the silu_mul kernel can use this to bound its
  // ROWS_PER_CTA loop instead of always processing all BM_PADDING (=128)
  // rows. Layout in row 1 of meta (after active_expert_mask):
  //
  //   meta[META_ROW_STRIDE       : META_ROW_STRIDE +   E_LOCAL] = mask
  //   meta[META_ROW_STRIDE+E_LOC : META_ROW_STRIDE + 2*E_LOCAL] = count
  if (tid == 0) {
    active_expert_mask[my_expert] = (actual_count > 0) ? 1 : 0;
    active_expert_mask[E_LOCAL + my_expert] = actual_count;
  }
}

} // namespace kernel
