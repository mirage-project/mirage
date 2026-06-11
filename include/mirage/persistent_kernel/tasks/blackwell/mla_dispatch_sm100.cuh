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

// MLA Dispatch Header for SM100
// Dispatches to the right kernel variant based on compile-time shape.
//
// Currently supported: DeepSeek V3 MLA
//   NUM_HEADS=128, D_K=576, D_V=512, TILE_S=128, 1 tile per split
//
// To add a new shape: add a new template specialization.

#pragma once

#include "mla_decode_sm100.cuh"

namespace kernel {

template <int>
struct mla_always_false {
  static constexpr bool value = false;
};

// Primary template: compile-time error for unsupported shapes.
template <int NUM_HEADS, int D_K, int D_V>
__device__ __forceinline__ void
    mla_decode_dispatch(CUtensorMap const *Q_tm_ptr,
                        CUtensorMap const *KV_tm_ptr,
                        float *Oa,
                        float *La,
                        float softmax_scale,
                        int kv_len,
                        int sk,
                        int split_idx,
                        int batch_idx) {
  static_assert(mla_always_false<NUM_HEADS>::value,
                "MLA decode: unsupported (NUM_HEADS, D_K, D_V) config");
}

// DeepSeek V3: H=128, D_K=576, D_V=512
template <>
__device__ __forceinline__ void
    mla_decode_dispatch<128, 576, 512>(CUtensorMap const *Q_tm_ptr,
                                       CUtensorMap const *KV_tm_ptr,
                                       float *Oa,
                                       float *La,
                                       float softmax_scale,
                                       int kv_len,
                                       int sk,
                                       int split_idx,
                                       int batch_idx) {
  mla_decode_sm100_task_impl(Q_tm_ptr,
                             KV_tm_ptr,
                             Oa,
                             La,
                             softmax_scale,
                             kv_len,
                             sk,
                             split_idx,
                             batch_idx);
}

} // namespace kernel
