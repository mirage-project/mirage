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
// this layout is used for tma, where inner col has stride 1, and
// outer col has stride ROW_ * INNER_COL_, row has stride INNER_COL_
template <typename T,
          int B,
          int M,
          int S,
          size_t ROW_,
          size_t INNER_COL_,
          size_t OUTER_COL_>
struct smem_tma {
  T *__restrict__ base_ptr;
  using value_type = T;
  static constexpr size_t ROW = ROW_;
  static constexpr size_t INNER_COL = INNER_COL_;
  static constexpr size_t OUTER_COL = OUTER_COL_;
  static constexpr size_t COL = INNER_COL * OUTER_COL;

  static constexpr size_t STRIDE_ROW = INNER_COL_;
  static constexpr size_t STRIDE_OUTER_COL = ROW_ * INNER_COL_;

  static constexpr int b = B;
  static constexpr int m = M;
  static constexpr int s = S;

  __device__ __forceinline__ smem_tma(T *ptr) : base_ptr(ptr) {}

  __device__ __forceinline__ void set_ptr(T *ptr) {
    base_ptr = ptr;
  }

  static constexpr size_t size() {
    return ROW * INNER_COL * OUTER_COL;
  }

  __device__ __forceinline__ size_t
      get_swizzled_offset(size_t logical_idx) const {
    // Skip swizzling calculation for B == 0
    if constexpr (B == 0) {
      return logical_idx;
    } else {
      size_t block_idx = logical_idx >> (M + S + B);
      size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);

      size_t irow = in_block_idx >> (M + S);
      size_t icol = (in_block_idx >> M) & ((1 << S) - 1);
      icol ^= irow;
      size_t offset_in_bank = in_block_idx & ((1 << M) - 1);
      size_t phy_offset = (block_idx << (M + S + B)) + (irow << (M + S)) +
                          (icol << M) + offset_in_bank;
#ifdef MIRAGE_DEBUG_HOPPER
      if (logical_idx == 0 && threadIdx.x == 0) {
        printf("block_idx: %llu\n", block_idx);
        printf("in_block_idx: %llu\n", in_block_idx);
        printf("irow: %llu\n", irow);
        printf("icol: %llu\n", icol);
        printf("offset_in_bank: %llu\n", offset_in_bank);
        printf("phy_offset: %llu\n", phy_offset);
        printf("base_ptr[logical_idx]: %f\n", (float)base_ptr[logical_idx]);
        printf("base_ptr[phy_offset]: %f\n", (float)base_ptr[phy_offset]);
      }
#endif
      return phy_offset;
    }
  }
  // 3D access
  __device__ __forceinline__ T *operator()(size_t logical_idx_row,
                                           size_t logical_idx_inner_col,
                                           size_t logical_idx_outer_col) {
    size_t logical_idx = logical_idx_row * STRIDE_ROW +
                         logical_idx_outer_col * STRIDE_OUTER_COL +
                         logical_idx_inner_col;

    return &base_ptr[get_swizzled_offset(logical_idx)];
  }
  __device__ __forceinline__ T &at(size_t logical_idx_row,
                                   size_t logical_idx_inner_col,
                                   size_t logical_idx_outer_col) {
    size_t logical_idx = logical_idx_row * STRIDE_ROW +
                         logical_idx_outer_col * STRIDE_OUTER_COL +
                         logical_idx_inner_col;
    return base_ptr[get_swizzled_offset(logical_idx)];
  }

  // 2D access
  __device__ __forceinline__ T *operator()(size_t logical_idx_row,
                                           size_t logical_idx_col) {

    size_t logical_idx_outer_col = logical_idx_col / INNER_COL;
    size_t logical_idx_inner_col = logical_idx_col % INNER_COL;
    size_t logical_idx = logical_idx_row * STRIDE_ROW +
                         logical_idx_outer_col * STRIDE_OUTER_COL +
                         logical_idx_inner_col;
    return &base_ptr[get_swizzled_offset(logical_idx)];
  }
  __device__ __forceinline__ T &at(size_t logical_idx_row,
                                   size_t logical_idx_col) {
    size_t logical_idx_outer_col = logical_idx_col / INNER_COL;
    size_t logical_idx_inner_col = logical_idx_col % INNER_COL;
    size_t logical_idx = logical_idx_row * STRIDE_ROW +
                         logical_idx_outer_col * STRIDE_OUTER_COL +
                         logical_idx_inner_col;
    return base_ptr[get_swizzled_offset(logical_idx)];
  }

  // 1D access
  __device__ __forceinline__ T &at(size_t logical_idx) {
    return base_ptr[get_swizzled_offset(logical_idx)];
  }

  __device__ __forceinline__ T *operator[](size_t logical_idx) const {
    return &base_ptr[get_swizzled_offset(logical_idx)];
  }
};
} // namespace kernel