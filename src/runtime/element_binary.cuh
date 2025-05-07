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

template <typename SMEM_DST, typename SMEM_SRC0, typename SMEM_SRC1>
static __device__ __forceinline__ void div_col(
    SMEM_DST dst,
    const SMEM_SRC0 src0,
    const SMEM_SRC1 src1) {

  static_assert(SMEM_SRC0::ROW == 1 && SMEM_SRC1::ROW == 1);
  static_assert(SMEM_SRC0::COL % SMEM_SRC1::COL == 0);

  constexpr int BLOCK_SIZE = SMEM_SRC0::COL / SMEM_SRC1::COL;

  for (int elem_idx = threadIdx.x; elem_idx < SMEM_DST::size(); elem_idx += NUM_THREADS) {
    int col = elem_idx % SMEM_DST::COL;
    int block_id = col / BLOCK_SIZE;
    float divisor = static_cast<float>(src1[0, block_id]);
    dst[0, col] = static_cast<float>(src0[0, col]) / divisor;
  }
}