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
#include "tasks/common/utils.cuh"
#pragma once

namespace kernel {

template <typename T,
          int B,
          int M,
          int S,
          size_t ROW_,
          size_t COL_,
          size_t STRIDE_>
struct smem_row_tiled {
  T *__restrict__ base_ptr;

  // Problem sizes
  static constexpr uint32_t ROW = ROW_;
  static constexpr uint32_t COL = COL_;
  static constexpr uint32_t STRIDE = STRIDE_;
  static constexpr uint32_t STRIDE_B128 = static_cast<uint32_t>(STRIDE / 8);

  // Fixed atom = 8 x 64
  static constexpr uint32_t AtomRBits = 3;                      // 8
  static constexpr uint32_t AtomCBits = 6;                      // 64
  static constexpr uint32_t AtomBits = AtomRBits + AtomCBits;   // 9
  static constexpr uint32_t AtomRMask = (1u << AtomRBits) - 1u; // 0b111
  static constexpr uint32_t AtomCMask = (1u << AtomCBits) - 1u; // 0b11_1111

  static constexpr uint32_t ColTiles = COL >> AtomCBits;

  static constexpr uint32_t MASK_B = (1u << B) - 1u;
  static constexpr uint32_t M_PLUS_S = M + S;

  static_assert(B <= AtomRBits, "B must fit in the 3 row-in-atom bits");
  static_assert(M_PLUS_S <= AtomCBits,
                "M+S must not exceed 6 for 64-col atoms");

  __device__ __forceinline__ smem_row_tiled(T *p) : base_ptr(p) {}
  __device__ __forceinline__ void set_ptr(T *p) {
    base_ptr = p;
  }

  __device__ __forceinline__ uint4 *get_128B_aligned_ptr(uint32_t row,
                                                         uint32_t col) {
    return reinterpret_cast<uint4 *>(base_ptr) + row * STRIDE_B128 +
           (col ^ (row & AtomRMask));
  }

  __device__ __forceinline__ T *operator()(uint32_t row, uint32_t col) const {
    const uint32_t r_outer = row >> AtomRBits;          // /8
    const uint32_t c_outer = col >> AtomCBits;          // /64
    const uint32_t tile = r_outer * ColTiles + c_outer; //

    const uint32_t r_in = row & AtomRMask; // 0..7
    const uint32_t c_in = col & AtomCMask; // 0..63

    const uint32_t c_swz = c_in ^ ((r_in & MASK_B) << M);

    const uint32_t idx = (tile << AtomBits) | (r_in << AtomCBits) | c_swz;

    return base_ptr + idx;
  }

  __device__ __forceinline__ T &at(uint32_t r, uint32_t c) {
    return *(*this)(r, c);
  }
};

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

  static constexpr size_t MASK_B = (1 << B_BITS) - 1;
  static constexpr size_t MASK_YYY = MASK_B << (M_BITS + S_BITS);

  __device__ __forceinline__ static size_t get_phy_offset(size_t logical_idx) {
    // refer to
    // https://github.com/NVIDIA/cutlass/blob/e6e2cc29f5e7611dfc6af0ed6409209df0068cf2/include/cute/swizzle.hpp#L76-L79.
    return logical_idx ^ ((logical_idx & MASK_YYY) >> S_BITS);
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
struct smem_row_2drow {
  T *__restrict__ base_ptr;

  using value_type = T;
  using OffsetCalculator = SwizzleOffsetCalculator<B, M, S>;

  static constexpr size_t ROW = ROW_;
  static constexpr size_t INNER_COL = INNER_COL_;
  static constexpr size_t log2_INNER_COL = log2_constexpr(INNER_COL_);
  static constexpr size_t OUTER_COL = OUTER_COL_;
  static constexpr size_t COL = INNER_COL * OUTER_COL;
  static constexpr size_t SIZE = ROW * COL;
  static constexpr size_t STRIDE_OUTER_COL = INNER_COL;
  static constexpr size_t STRIDE = INNER_COL * OUTER_COL;

  __device__ __forceinline__ smem_row_2drow(T *ptr) : base_ptr(ptr) {}

  __device__ __forceinline__ void set_ptr(T *ptr) {
    base_ptr = ptr;
  }

  static constexpr size_t size() {
    return ROW * COL;
  }

  __device__ __forceinline__ T &at(size_t logical_idx_row,
                                   size_t logical_idx_col) {
    // size_t inner_col = logical_idx_col & ((1 << log2_INNER_COL) - 1);
    // size_t outer_col = logical_idx_col >> log2_INNER_COL;

    size_t inner_col = (logical_idx_col % INNER_COL);
    size_t outer_col = (logical_idx_col / INNER_COL) % OUTER_COL;

    size_t logical_idx =
        outer_col * STRIDE_OUTER_COL + logical_idx_row * STRIDE + inner_col;
    // return &base_ptr[get_swizzled_offset(logical_idx)];
    return base_ptr[OffsetCalculator::get_phy_offset(logical_idx)];
  }

  __device__ __forceinline__ T *operator()(size_t logical_idx_row,
                                           size_t logical_idx_col) {
    // size_t inner_col = logical_idx_col & ((1 << log2_INNER_COL) - 1);
    // size_t outer_col = logical_idx_col >> log2_INNER_COL;

    size_t inner_col = (logical_idx_col % INNER_COL);
    size_t outer_col = (logical_idx_col / INNER_COL) % OUTER_COL;

    size_t logical_idx =
        outer_col * STRIDE_OUTER_COL + logical_idx_row * STRIDE + inner_col;
    // return &base_ptr[get_swizzled_offset(logical_idx)];
    return &base_ptr[OffsetCalculator::get_phy_offset(logical_idx)];
  }
};

// Row-major layout with split column dimension: OUTER_COL x INNER_COL
template <typename T,
          int B,
          int M,
          int S,
          size_t ROW_,
          size_t COL_,
          size_t STAGE_>
struct smem_row_2dcol {
  T *__restrict__ base_ptr;

  using value_type = T;
  using OffsetCalculator = SwizzleOffsetCalculator<B, M, S>;

  static constexpr size_t ROW = ROW_;
  static constexpr size_t INNER_COL = 128 / sizeof(T);
  static constexpr size_t log2_INNER_COL = log2_constexpr(INNER_COL);
  static constexpr size_t OUTER_COL = COL_ / INNER_COL;
  static constexpr size_t COL = COL_;
  static constexpr size_t SIZE = ROW * COL;
  static constexpr size_t STAGE = STAGE_;
  static constexpr size_t STRIDE_OUTER_COL = ROW * INNER_COL;
  static constexpr size_t STRIDE_ROW =
      INNER_COL; // a row just contains inner col elements
  static constexpr size_t STRIDE_STAGE = SIZE;

  __device__ __forceinline__ smem_row_2dcol(T *ptr) : base_ptr(ptr) {}

  __device__ __forceinline__ void set_ptr(T *ptr) {
    base_ptr = ptr;
  }

  __device__ __forceinline__ T *
      operator()(size_t logical_idx_row, size_t logical_idx_col, size_t stage) {
    // stage was in row dim
    size_t inner_col = logical_idx_col & ((1 << log2_INNER_COL) - 1);
    size_t outer_col = logical_idx_col >> log2_INNER_COL;
    size_t logical_idx = stage * STRIDE_STAGE + outer_col * STRIDE_OUTER_COL +
                         logical_idx_row * STRIDE_ROW + inner_col;
    return &base_ptr[OffsetCalculator::get_phy_offset(logical_idx)];
  }

  __device__ __forceinline__ T &at(size_t logical_idx_row,
                                   size_t logical_idx_col) {
    size_t inner_col = logical_idx_col & ((1 << log2_INNER_COL) - 1);
    size_t outer_col = logical_idx_col >> log2_INNER_COL;
    size_t logical_idx =
        outer_col * STRIDE_OUTER_COL + logical_idx_row * STRIDE_ROW + inner_col;
    // return &base_ptr[get_swizzled_offset(logical_idx)];
    return base_ptr[OffsetCalculator::get_phy_offset(logical_idx)];
  }
};

// Column-major layout with split row dimension: OUTER_ROW x INNER_ROW
template <typename T,
          int B,
          int M,
          int S,
          size_t ROW_,
          size_t COL_,
          size_t STAGE_>
struct smem_col_2drow {
  T *__restrict__ base_ptr;

  using value_type = T;
  using OffsetCalculator = SwizzleOffsetCalculator<B, M, S>;

  static constexpr size_t COL = COL_;
  static constexpr size_t INNER_ROW = 128 / sizeof(T);
  static constexpr size_t log2_INNER_ROW = log2_constexpr(INNER_ROW);
  static constexpr size_t OUTER_ROW = ROW_ / INNER_ROW;
  static constexpr size_t ROW = ROW_;
  static constexpr size_t SIZE = ROW * COL;
  static constexpr size_t STAGE = STAGE_;
  static constexpr size_t STRIDE_OUTER_ROW = COL * INNER_ROW;
  static constexpr size_t STRIDE_COL =
      INNER_ROW; // a col just contains inner row elements
  static constexpr size_t STRIDE_STAGE = SIZE;

  static constexpr size_t INNER_ROW_MASK = (1 << log2_INNER_ROW) - 1;

  __device__ __forceinline__ smem_col_2drow(T *ptr) : base_ptr(ptr) {}

  __device__ __forceinline__ void set_ptr(T *ptr) {
    base_ptr = ptr;
  }

  __device__ __forceinline__ size_t get_logical_idx(size_t logical_idx_row,
                                                    size_t logical_idx_col,
                                                    size_t stage) {
    size_t inner_row = logical_idx_row & INNER_ROW_MASK;
    size_t outer_row = logical_idx_row >> log2_INNER_ROW;
    size_t logical_idx = stage * STRIDE_STAGE + outer_row * STRIDE_OUTER_ROW +
                         logical_idx_col * STRIDE_COL + inner_row;
    return logical_idx;
  }

  __device__ __forceinline__ T *
      operator()(size_t logical_idx_row, size_t logical_idx_col, size_t stage) {
    // stage was in col dim
    size_t logical_idx =
        get_logical_idx(logical_idx_row, logical_idx_col, stage);
    return &base_ptr[OffsetCalculator::get_phy_offset(logical_idx)];
  }
};

} // namespace kernel
