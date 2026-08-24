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
#include "runtime_header.h"

namespace kernel {

// How many KV tokens a paged-attention kernel loads per iteration.
//
// These used to be function-local constexprs, one per kernel, so nothing
// outside a kernel could see the value it tiles at. That matters because the
// scheduler and the kernel independently compute where a sliding window
// starts, and they must agree exactly:
//
//   kernel     first_kv_iter    = max(seq_len - num_tokens - WINDOW + 1, 0)
//                                 / KV_TILE_SIZE
//   scheduler  first_live_token = (max(step - WINDOW + 1, 0)
//                                 / MPK_KV_WINDOW_TILE) * MPK_KV_WINDOW_TILE
//
// If the scheduler's tile is LARGER than the kernel's, prepare_next_batch
// frees pages the kernel is still reading — silent corruption. Smaller is
// merely wasteful. Hoisting the constants here is what lets the windowed
// kernels static_assert that correspondence.
constexpr int KV_TILE_SM100 = 64;        // attention_sm100, dflash_sm100, tma
constexpr int KV_TILE_HOPPER = 64;       // multitoken_paged_attention_hopper
constexpr int KV_TILE_AMPERE_4_16 = 64;  // multitoken_paged_attention_4_16{,_split_kv}
constexpr int KV_TILE_AMPERE_32_64 = 128; // multitoken_paged_attention_32_64{,_split_kv}

// The granularity prepare_next_batch frees a sliding window's pages at. Every
// kernel that supports a window must tile at exactly this; the 128-token
// Ampere variants may not grow one without changing this too.
constexpr int KV_WINDOW_TILE = MPK_KV_WINDOW_TILE;

} // namespace kernel
