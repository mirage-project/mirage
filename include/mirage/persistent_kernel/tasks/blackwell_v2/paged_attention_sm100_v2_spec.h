/* Copyright 2026 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */
#pragma once

// Single source of truth for the paged_attention_sm100_v2 SMEM region
// declaration fed to the planner.
//
// attention_sm100.cuh addresses SMEM via AttentionBuffersImpl(char *smem)
// with hand-rolled per-buffer offsets, and does NOT call
// task_desc->smem_region_offset(...). To keep the planner's page assignment
// truthful about which bytes the device touches, we declare a SINGLE
// monolithic region: the planner allocates one contiguous run of pages
// starting at some page_start, the device writes bytes [0, total_bytes)
// inside that run via its internal offsets, and the page IDs the planner
// assigns are exactly the ones the data lives on.
//
// FUTURE: split back into per-buffer regions and rewrite attention to read
// smem_region_offset(N) per buffer. That restores the per-buffer release_step
// granularity {1, 2, 3} that was originally declared. For now we trade that
// granularity for correctness.
//
// Host-safe.

#include "mirage/kernel/task_register.h"

namespace kernel {
namespace paged_attention_sm100_v2 {

// Mirrors AttentionBuffersImpl in attention_sm100.cuh. Sizes depend on the
// runtime instantiation (head_dim, num_q_heads, num_kv_heads).
struct Layout {
  int total_bytes;
  int alignment;
  int zero_b;
  int qo_b;
  int kv_b;
  int norm_b;
  int md_b;
  int obuf_b;
};

inline constexpr int round_up(int n, int a) {
  return (n + a - 1) & ~(a - 1);
}

inline Layout compute_layout(int num_q_heads, int num_kv_heads, int head_dim) {
  // These constants must mirror the device-side defaults in
  // multitoken_paged_attention_sm100_task_impl. Keep in sync.
  int const max_tokens   = 8;
  int const kv_tile      = 64;
  int const num_threads  = 128;
  int const t_size       = 2;  // bf16
  int const qo_per_kv    = num_q_heads / num_kv_heads;
  int const mma_iters_m  = (max_tokens * qo_per_kv + 15) / 16;

  Layout L;
  L.zero_b = t_size * 8;
  L.qo_b   = t_size * max_tokens * qo_per_kv * head_dim;
  L.kv_b   = t_size * kv_tile * head_dim;
  L.norm_b = (int)sizeof(float) * 4;
  L.md_b   = (int)sizeof(float) * mma_iters_m * num_threads * 2;
  L.obuf_b = (int)sizeof(float) * mma_iters_m * num_threads * 64;

  int const zero_off         = 0;
  int const s_q_off          = zero_off + L.zero_b;
  int const s_k_off          = s_q_off + L.qo_b;
  int const s_k_buffer_off   = s_k_off + L.kv_b;
  int const s_v_off          = s_k_buffer_off + L.kv_b;
  int const s_v_buffer_off   = s_v_off + L.kv_b;
  int const s_o_off          = s_v_buffer_off + L.kv_b;
  int const s_q_norm_sum_off = round_up(s_o_off + L.qo_b, (int)sizeof(float));
  int const s_k_norm_sum_off = s_q_norm_sum_off + L.norm_b;
  int const s_m_buffer_off   = s_k_norm_sum_off + L.norm_b;
  int const s_d_buffer_off   = s_m_buffer_off + L.md_b;
  int const s_o_buffer_off   = s_d_buffer_off + L.md_b;
  L.total_bytes              = s_o_buffer_off + L.obuf_b;
  L.alignment                = 1024;
  return L;
}

inline ::mirage::runtime::TaskSmemInfo
make_smem_info(int num_q_heads, int num_kv_heads, int head_dim) {
  Layout L = compute_layout(num_q_heads, num_kv_heads, head_dim);
  ::mirage::runtime::TaskSmemInfo info{L.total_bytes, L.alignment, {}};
  // Single monolithic region. Per-buffer release_step granularity {1,2,3}
  // that was originally declared is gone; see file header comment.
  info.regions.push_back({"attention_smem", L.total_bytes, L.alignment,
                          /*page_count=*/-1, /*can_pack=*/false,
                          /*release_step=*/1, /*contiguous=*/true});
  return info;
}

} // namespace paged_attention_sm100_v2
} // namespace kernel
