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

 namespace mirage {
     namespace runtime {

constexpr int NUM_THREADS = 128;
        //reduction on dim 0
 template <typename SMEM_DST, typename SMEM_SRC>
static __device__ __forceinline__ void reduction_sum_row(SMEM_DST dst,
    const SMEM_SRC src) {

static constexpr int REDUCTION_FACTOR =  SMEM_SRC::ROW
for (int dst_elem_idx = threadIdx.x; dst_elem_idx < SMEM_DST::size();
            dst_elem_idx += NUM_THREADS) {
    float result = 0;
    int dst_col = dst_elem_idx % SMEM_DST::COL;

    #pragma unroll
    for (int i = 0; i < REDUCTION_FACTOR; ++i) {
        result += static_cast<float>(src[i, dst_col]);
    }
    dst[dst_elem_idx] = result;
}
}

//reduction on dim 1
template <typename SMEM_DST, typename SMEM_SRC>
static __device__ __forceinline__ void reduction_sum_col(SMEM_DST dst,
    const SMEM_SRC src) {
static constexpr int REDUCTION_FACTOR = SMEM_SRC::COL;
for (int dst_elem_idx = threadIdx.x; dst_elem_idx < SMEM_DST::size();
            dst_elem_idx += NUM_THREADS) {
    float result = 0;
    int dst_row = dst_elem_idx / SMEM_DST::COL;
    #pragma unroll
    for (int i = 0; i < REDUCTION_FACTOR; ++i) {
            result += static_cast<float>(src[dst_row, i]);

    }
    dst[dst_elem_idx] = result;
}
}

}}
