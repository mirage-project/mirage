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
  // 2D access
  __device__ __forceinline__ T *operator()(size_t logical_idx_row,
                                           size_t logical_idx_col) {
    size_t logical_idx = logical_idx_row * STRIDE + logical_idx_col;
    // assert(logical_idx < SIZE);

    // first get the block -> 3,3,3 means 8*8*8 = 8 rows * 8 cols * 8elements
    // size_t block_idx = logical_idx / (Pow2_M * Pow2_S * Pow2_B);
    // size_t in_block_idx = logical_idx % (Pow2_M * Pow2_S * Pow2_B);

    // //get the irow and icol inside the block
    // size_t irow = in_block_idx / (Pow2_M * Pow2_S);
    // size_t icol = (in_block_idx / (Pow2_M)) % (Pow2_S);
    // icol = irow ^ icol;
    // size_t offset_in_bank = in_block_idx % (Pow2_M);

    // size_t phy_offset = block_idx * (Pow2_M * Pow2_S * Pow2_B) + irow *
    // (Pow2_M * Pow2_S)
    // + icol * (Pow2_M) + offset_in_bank;

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
} // namespace kernel