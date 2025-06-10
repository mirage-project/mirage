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
template <typename T, size_t ROW, size_t COL, size_t STRIDE>
struct dmem_row {
  T *base_ptr;

  __device__ __forceinline__ dmem_row(T *ptr) : base_ptr(ptr) {}

  __device__ __forceinline__ void set_ptr(T *ptr) {
    base_ptr = ptr;
  }

  __device__ __forceinline__ T *operator()(size_t logical_idx_row,
                                           size_t logical_idx_col) {
    size_t logical_idx = logical_idx_row * STRIDE + logical_idx_col;
    // assert(logical_idx < (ROW*COL));
    return &base_ptr[logical_idx];
  }

  __device__ __forceinline__ T &at(size_t logical_idx) {
    // assert(logical_idx < (ROW*COL));
    return base_ptr[logical_idx];
  }
  __device__ __forceinline__ T &at(size_t logical_idx_row,
                                   size_t logical_idx_col) {

    size_t logical_idx = logical_idx_row * STRIDE + logical_idx_col;
    // assert(logical_idx < (ROW*COL));
    return base_ptr[logical_idx];
  }
};

template <typename T, size_t ROW, size_t COL, size_t STRIDE>
struct dmem_row_const {
  T const *base_ptr;

  __device__ __forceinline__ dmem_row_const(T const *ptr) : base_ptr(ptr) {}

  __device__ __forceinline__ void set_ptr(T const *ptr) {
    base_ptr = ptr;
  }

  __device__ __forceinline__ T const *operator()(size_t logical_idx_row,
                                                 size_t logical_idx_col) const {
    size_t logical_idx = logical_idx_row * STRIDE + logical_idx_col;
    // assert(logical_idx < (ROW*COL));
    return &base_ptr[logical_idx];
  }

  __device__ __forceinline__ T const &at(size_t logical_idx) const {
    // assert(logical_idx < (ROW*COL));
    return base_ptr[logical_idx];
  }
  __device__ __forceinline__ T const &at(size_t logical_idx_row,
                                         size_t logical_idx_col) const {
    size_t logical_idx = logical_idx_row * STRIDE + logical_idx_col;
    // assert(logical_idx < (ROW*COL));
    return base_ptr[logical_idx];
  }
};

template <typename T, size_t ROW, size_t COL, size_t STRIDE>
struct dmem_col_const {
  T const *base_ptr;

  __device__ __forceinline__ dmem_col_const(T const *ptr) : base_ptr(ptr) {}

  __device__ __forceinline__ void set_ptr(T const *ptr) {
    base_ptr = ptr;
  }

  __device__ __forceinline__ T const *operator()(size_t logical_idx_row,
                                                 size_t logical_idx_col) const {
    size_t logical_idx = logical_idx_col * STRIDE + logical_idx_row;
    return &base_ptr[logical_idx];
  }

  __device__ __forceinline__ T const &at(size_t logical_idx) const {
    return base_ptr[logical_idx];
  }
  __device__ __forceinline__ T const &at(size_t logical_idx_row,
                                         size_t logical_idx_col) const {
    size_t logical_idx = logical_idx_col * STRIDE + logical_idx_row;
    return base_ptr[logical_idx];
  }
};
} // namespace kernel