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
#pragma once
#include "runtime_header.h"

namespace kernel {

// How many KV tokens a paged-attention kernel loads per iteration.
constexpr int KV_TILE_SM100 = 64;        // attention_sm100, dflash_sm100, tma
constexpr int KV_TILE_HOPPER = 64;       // multitoken_paged_attention_hopper
constexpr int KV_TILE_AMPERE_4_16 = 64;  // multitoken_paged_attention_4_16{,_split_kv}
constexpr int KV_TILE_AMPERE_32_64 = 128; // multitoken_paged_attention_32_64{,_split_kv}

// The granularity prepare_next_batch frees a sliding window's pages at. Every
// kernel that supports a window must tile at exactly this; the 128-token
// Ampere variants may not grow one without changing this too.
constexpr int KV_WINDOW_TILE = MPK_KV_WINDOW_TILE;

} // namespace kernel
