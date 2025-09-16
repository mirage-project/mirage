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
#include "utils.cuh"
#pragma once
namespace kernel {
template <typename T,
          int B,
          int M,
          int S,
          size_t ROW_,
          size_t COL_,
          size_t STRIDE>
struct smem_row {
  T *__restrict__ base_ptr;
  using value_type = T;
  static constexpr int b = B;
  static constexpr int m = M;
  static constexpr int s = S;

  static constexpr size_t ROW = ROW_;
  static constexpr size_t COL = COL_;
  static constexpr size_t SIZE = ROW * COL;

  __device__ __forceinline__ smem_row(T *ptr) : base_ptr(ptr) {}

  __device__ __forceinline__ void set_ptr(T *ptr) {
    base_ptr = ptr;
  }

  static constexpr size_t size() {
    return ROW * COL;
  }
  __device__ __forceinline__ int get_offset_in_bank(size_t logical_idx_row,
                                                    size_t logical_idx_col) {
    size_t logical_idx = logical_idx_row * STRIDE + logical_idx_col;
    // assert(logical_idx < SIZE);
    size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);
    size_t offset_in_bank = in_block_idx & ((1 << M) - 1);
    return offset_in_bank;
  }
  __device__ __forceinline__ int get_phy_offset(size_t logical_idx_row,
                                                size_t logical_idx_col) {
    size_t logical_idx = logical_idx_row * STRIDE + logical_idx_col;
    // assert(logical_idx < SIZE);

    size_t block_idx = logical_idx >> (M + S + B);
    size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);

    size_t irow = in_block_idx >> (M + S);
    size_t icol = (in_block_idx >> M) & ((1 << S) - 1);
    icol ^= irow;
    size_t offset_in_bank = in_block_idx & ((1 << M) - 1);
    size_t phy_offset = (block_idx << (M + S + B)) + (irow << (M + S)) +
                        (icol << M) + offset_in_bank;
    return phy_offset;
  }
  // 2D access
  __device__ __forceinline__ T *operator()(size_t logical_idx_row,
                                           size_t logical_idx_col) {
    size_t logical_idx = logical_idx_row * STRIDE + logical_idx_col;
    // assert(logical_idx < SIZE);

    size_t block_idx = logical_idx >> (M + S + B);
    size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);

    size_t irow = in_block_idx >> (M + S);
    size_t icol = (in_block_idx >> M) & ((1 << S) - 1);
    icol ^= irow;
    size_t offset_in_bank = in_block_idx & ((1 << M) - 1);
    size_t phy_offset = (block_idx << (M + S + B)) + (irow << (M + S)) +
                        (icol << M) + offset_in_bank;
    // printf("phy_offset %d, %d\n", (int)logical_idx, (int)phy_offset);
    return &base_ptr[phy_offset];
  }
  __device__ __forceinline__ T &at(size_t logical_idx_row,
                                   size_t logical_idx_col) {
    size_t logical_idx = logical_idx_row * STRIDE + logical_idx_col;
    // assert(logical_idx < SIZE);

    size_t block_idx = logical_idx >> (M + S + B);
    size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);

    size_t irow = in_block_idx >> (M + S);
    size_t icol = (in_block_idx >> M) & ((1 << S) - 1);
    icol ^= irow;
    size_t offset_in_bank = in_block_idx & ((1 << M) - 1);

    size_t phy_offset = (block_idx << (M + S + B)) + (irow << (M + S)) +
                        (icol << M) + offset_in_bank;
    return base_ptr[phy_offset];
  }

  // 1D access
  __device__ __forceinline__ T &at(size_t logical_idx) {
    // assert(logical_idx < SIZE);
    size_t block_idx = logical_idx >> (M + S + B);
    size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);

    size_t irow = in_block_idx >> (M + S);
    size_t icol = (in_block_idx >> M) & ((1 << S) - 1);
    icol ^= irow;
    size_t offset_in_bank = in_block_idx & ((1 << M) - 1);

    size_t phy_offset = (block_idx << (M + S + B)) + (irow << (M + S)) +
                        (icol << M) + offset_in_bank;
    return base_ptr[phy_offset];
  }

  __device__ __forceinline__ T *operator[](size_t logical_idx) const {
    // assert(logical_idx < SIZE);
    size_t block_idx = logical_idx >> (M + S + B);
    size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);

    size_t irow = in_block_idx >> (M + S);
    size_t icol = (in_block_idx >> M) & ((1 << S) - 1);
    icol ^= irow;
    size_t offset_in_bank = in_block_idx & ((1 << M) - 1);

    size_t phy_offset = (block_idx << (M + S + B)) + (irow << (M + S)) +
                        (icol << M) + offset_in_bank;
    return &base_ptr[phy_offset];
  }
};

template <typename T,
          int B,
          int M,
          int S,
          size_t ROW_,
          size_t COL_,
          size_t STRIDE>
struct smem_col {
  T *base_ptr;

  using value_type = T;

  static constexpr int b = B;
  static constexpr int m = M;
  static constexpr int s = S;

  static constexpr size_t ROW = ROW_;
  static constexpr size_t COL = COL_;

  __device__ __forceinline__ smem_col(T *ptr) : base_ptr(ptr) {}

  __device__ __forceinline__ void set_ptr(T *ptr) {
    base_ptr = ptr;
  }

  __device__ __forceinline__ int get_original_icol(size_t logical_idx_row,
                                                   size_t logical_idx_col) {
    size_t logical_idx = logical_idx_col * STRIDE + logical_idx_row;
    // assert(logical_idx < SIZE);

    size_t block_idx = logical_idx >> (M + S + B);
    size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);

    size_t irow = in_block_idx >> (M + S);
    size_t icol = (in_block_idx >> M) & ((1 << S) - 1);
    return icol;
  }

  __device__ __forceinline__ int get_swizzled_icol(size_t logical_idx_row,
                                                   size_t logical_idx_col) {
    size_t logical_idx = logical_idx_col * STRIDE + logical_idx_row;
    // assert(logical_idx < SIZE);

    size_t block_idx = logical_idx >> (M + S + B);
    size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);

    size_t irow = in_block_idx >> (M + S);
    size_t icol = (in_block_idx >> M) & ((1 << S) - 1);
    icol ^= irow;
    return icol;
  }

  __device__ __forceinline__ int get_phy_offset(size_t logical_idx_row,
                                                size_t logical_idx_col) {
    size_t logical_idx = logical_idx_col * STRIDE + logical_idx_row;
    // assert(logical_idx < SIZE);

    size_t block_idx = logical_idx >> (M + S + B);
    size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);

    size_t irow = in_block_idx >> (M + S);
    size_t icol = (in_block_idx >> M) & ((1 << S) - 1);
    icol ^= irow;
    size_t offset_in_bank = in_block_idx & ((1 << M) - 1);

    size_t phy_offset = (block_idx << (M + S + B)) + (irow << (M + S)) +
                        (icol << M) + offset_in_bank;
    return phy_offset;
  }

  __device__ __forceinline__ T *operator()(size_t logical_idx_row,
                                           size_t logical_idx_col) {
    size_t logical_idx = logical_idx_col * STRIDE + logical_idx_row;
    size_t block_idx = logical_idx >> (M + S + B);
    size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);

    size_t irow = in_block_idx >> (M + S);
    size_t icol = (in_block_idx >> M) & ((1 << S) - 1);
    icol ^= irow;
    size_t offset_in_bank = in_block_idx & ((1 << M) - 1);

    size_t phy_offset = (block_idx << (M + S + B)) + (irow << (M + S)) +
                        (icol << M) + offset_in_bank;
    return &base_ptr[phy_offset];
  }

  __device__ __forceinline__ T &at(size_t logical_idx) {
    size_t block_idx = logical_idx >> (M + S + B);
    size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);

    size_t irow = in_block_idx >> (M + S);
    size_t icol = (in_block_idx >> M) & ((1 << S) - 1);
    icol ^= irow;
    size_t offset_in_bank = in_block_idx & ((1 << M) - 1);

    size_t phy_offset = (block_idx << (M + S + B)) + (irow << (M + S)) +
                        (icol << M) + offset_in_bank;
    return base_ptr[phy_offset];
  }

  __device__ __forceinline__ T &at(size_t logical_idx_row,
                                   size_t logical_idx_col) {
    size_t logical_idx = logical_idx_col * STRIDE + logical_idx_row;
    size_t block_idx = logical_idx >> (M + S + B);
    size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);

    size_t irow = in_block_idx >> (M + S);
    size_t icol = (in_block_idx >> M) & ((1 << S) - 1);
    icol ^= irow;
    size_t offset_in_bank = in_block_idx & ((1 << M) - 1);

    size_t phy_offset = (block_idx << (M + S + B)) + (irow << (M + S)) +
                        (icol << M) + offset_in_bank;
    return base_ptr[phy_offset];
  }

  __device__ __forceinline__ T *operator[](size_t logical_idx) const {
    size_t block_idx = logical_idx >> (M + S + B);
    size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);

    size_t irow = in_block_idx >> (M + S);
    size_t icol = (in_block_idx >> M) & ((1 << S) - 1);
    icol ^= irow;
    size_t offset_in_bank = in_block_idx & ((1 << M) - 1);

    size_t phy_offset = (block_idx << (M + S + B)) + (irow << (M + S)) +
                        (icol << M) + offset_in_bank;
    return &base_ptr[phy_offset];
  }
};

template <int B, int M, int S>
struct SwizzleOffsetCalculator {
  static constexpr size_t B_BITS = B;
  static constexpr size_t M_BITS = M;
  static constexpr size_t S_BITS = S;
  static constexpr size_t BLOCK_BITS = B + M + S;
  static constexpr size_t M_AND_S_BITS = M + S;

  static constexpr size_t MASK_IN_BLOCK = (1 << BLOCK_BITS) - 1;
  static constexpr size_t MASK_S = (1 << S_BITS) - 1;
  static constexpr size_t MASK_M = (1 << M_BITS) - 1;

  __device__ __forceinline__ static size_t get_phy_offset(size_t logical_idx) {
    const size_t block_idx = logical_idx >> BLOCK_BITS;
    const size_t in_block_idx = logical_idx & MASK_IN_BLOCK;

    const size_t irow = in_block_idx >> M_AND_S_BITS;
    size_t icol = (in_block_idx >> M_BITS) & MASK_S;
    icol ^= irow;

    const size_t offset_in_bank = in_block_idx & MASK_M;

    return (block_idx << BLOCK_BITS) + (irow << M_AND_S_BITS) +
           (icol << M_BITS) + offset_in_bank;
  }
};

// Row-major layout with split column dimension: OUTER_COL x INNER_COL
template <typename T,
          int B,
          int M,
          int S,
          size_t ROW_,
          size_t INNER_COL_,
          size_t OUTER_COL_>
struct smem_row_2dcol {
  T *__restrict__ base_ptr;

  using value_type = T;
  using OffsetCalculator = SwizzleOffsetCalculator<B, M, S>;

  static constexpr size_t ROW = ROW_;
  static constexpr size_t INNER_COL = INNER_COL_;
  static constexpr size_t log2_INNER_COL = log2_constexpr(INNER_COL_);
  static constexpr size_t OUTER_COL = OUTER_COL_;
  static constexpr size_t COL = INNER_COL * OUTER_COL;
  static constexpr size_t SIZE = ROW * COL;
  static constexpr size_t STRIDE_OUTER_COL = ROW * INNER_COL;
  static constexpr size_t STRIDE = INNER_COL;

  __device__ __forceinline__ smem_row_2dcol(T *ptr) : base_ptr(ptr) {}

  __device__ __forceinline__ void set_ptr(T *ptr) {
    base_ptr = ptr;
  }

  static constexpr size_t size() {
    return ROW * COL;
  }

  __device__ __forceinline__ int get_offset_in_bank(size_t logical_idx_row,
                                                    size_t logical_idx_col) {
    size_t inner_col = logical_idx_col & ((1 << log2_INNER_COL) - 1);
    size_t outer_col = logical_idx_col >> log2_INNER_COL;
    size_t logical_idx =
        outer_col * STRIDE_OUTER_COL + logical_idx_row * STRIDE + inner_col;
    size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);
    size_t offset_in_bank = in_block_idx & ((1 << M) - 1);
    return offset_in_bank;
  }

  __device__ __forceinline__ T *operator()(size_t logical_idx_row,
                                           size_t logical_idx_col) {
    size_t inner_col = logical_idx_col & ((1 << log2_INNER_COL) - 1);
    size_t outer_col = logical_idx_col >> log2_INNER_COL;
    size_t logical_idx =
        outer_col * STRIDE_OUTER_COL + logical_idx_row * STRIDE + inner_col;
    // return &base_ptr[get_swizzled_offset(logical_idx)];
    return &base_ptr[OffsetCalculator::get_phy_offset(logical_idx)];
  }
};

// Column-major layout with split row dimension: OUTER_ROW x INNER_ROW
template <typename T,
          int B,
          int M,
          int S,
          size_t INNER_ROW_,
          size_t OUTER_ROW_,
          size_t COL_>
struct smem_col_2drow {
  T *__restrict__ base_ptr;

  using value_type = T;
  using OffsetCalculator = SwizzleOffsetCalculator<B, M, S>;

  static constexpr size_t INNER_ROW = INNER_ROW_;
  static constexpr size_t log2_INNER_ROW = log2_constexpr(INNER_ROW_);
  static constexpr size_t OUTER_ROW = OUTER_ROW_;
  static constexpr size_t ROW = OUTER_ROW_ * INNER_ROW_;
  static constexpr size_t COL = COL_;
  static constexpr size_t SIZE = ROW * COL;
  static constexpr size_t STRIDE_OUTER_ROW = COL * INNER_ROW_;
  static constexpr size_t STRIDE = INNER_ROW_;

  static constexpr size_t INNER_ROW_MASK = (1 << log2_INNER_ROW) - 1;

  __device__ __forceinline__ smem_col_2drow(T *ptr) : base_ptr(ptr) {}

  __device__ __forceinline__ void set_ptr(T *ptr) {
    base_ptr = ptr;
  }

  static constexpr size_t size() {
    return ROW * COL;
  }

  __device__ __forceinline__ size_t get_swizzled_icol(size_t logical_idx) {
    size_t block_idx = logical_idx >> (M + S + B);
    size_t in_block_idx = logical_idx & ((1 << (M + S + B)) - 1);

    size_t irow = in_block_idx >> (M + S);
    size_t icol = (in_block_idx >> M) & ((1 << S) - 1);
    icol ^= irow;
    return icol;
  }

  __device__ __forceinline__ size_t get_logical_idx(size_t logical_idx_row,
                                                    size_t logical_idx_col) {
    size_t inner_row = logical_idx_row & INNER_ROW_MASK;
    size_t outer_row = logical_idx_row >> log2_INNER_ROW;
    size_t logical_idx =
        outer_row * STRIDE_OUTER_ROW + logical_idx_col * STRIDE + inner_row;
    return logical_idx;
  }

  __device__ __forceinline__ size_t get_bank_idx(size_t logical_idx_row,
                                                 size_t logical_idx_col) {
    size_t logical_idx = get_logical_idx(logical_idx_row, logical_idx_col);
    return get_swizzled_icol(logical_idx);
  }

  __device__ __forceinline__ T *operator()(size_t logical_idx_row,
                                           size_t logical_idx_col) {
    size_t logical_idx = get_logical_idx(logical_idx_row, logical_idx_col);
    return &base_ptr[OffsetCalculator::get_phy_offset(logical_idx)];
  }
};

} // namespace kernel
