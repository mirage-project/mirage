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
namespace kernel {

template <typename SMEM_DST, typename SMEM_SRC0, typename SMEM_SRC1>
static __device__ __forceinline__ void
    div_col(SMEM_DST dst, SMEM_SRC0 src0, SMEM_SRC1 src1) {

  static_assert(SMEM_SRC0::ROW == SMEM_SRC1::ROW);
  static_assert(SMEM_SRC0::COL % SMEM_SRC1::COL == 0);
  static_assert(SMEM_SRC1::COL == 1);
  constexpr int BLOCK_SIZE = SMEM_SRC0::COL / SMEM_SRC1::COL;

  for (int elem_idx = threadIdx.x; elem_idx < SMEM_DST::size();
       elem_idx += NUM_THREADS) {
    int col = elem_idx % SMEM_DST::COL;
    int row = elem_idx / SMEM_DST::COL;
    dst.at(row, col) = (src0.at(row, col)) / (src1.at(row, 0));
  }
}

template <typename SMEM_D, typename SMEM_A, typename SMEM_B>
__device__ __forceinline__ void mul(SMEM_D dst, SMEM_A srcA, SMEM_B srcB) {
  for (int elem_idx = threadIdx.x; elem_idx < SMEM_D::size();
       elem_idx += NUM_THREADS) {
    dst.at(elem_idx) = srcA.at(elem_idx) * srcB.at(elem_idx);
  }
}
} // namespace kernel