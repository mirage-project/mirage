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

template <typename T, int B, int M=3, int S, size_t ROW, size_t COL, size_t STRIDE>
struct smem_row {
    T* __restrict__ base_ptr;

    using value_type = T;

    static constexpr size_t Pow2_M = (1 << M);
    static constexpr size_t Pow2_S = (1 << S);
    static constexpr size_t Pow2_B = (1 << B);

    __device__ __forceinline__
    smem_row(T* ptr) : base_ptr(ptr) {}

    static constexpr size_t size() { return ROW * COL; }
    
    // 2D access
    __device__ __forceinline__
    T& operator[] (size_t logical_idx_row, size_t logical_idx_col) const {
        size_t logical_idx = logical_idx_row * STRIDE + logical_idx_col;
        //coordinate must be start of a bank
        assert(logical_idx % Pow2_M == 0);

        size_t row = logical_idx >> (S + S);
        size_t irow = row % Pow2_B;
        size_t icol = irow ^ (logical_idx >> M) % (S);
        size_t phy_offset = row << (M + S) + icol << M;
        return base_ptr[phy_offset];
    }

    // 1D access
    __device__ __forceinline__
    T& operator[] (size_t logical_idx) const {

        // assert(logical_idx % Pow2_M == 0);
        size_t offset_in_bank = logical_idx % Pow2_M;
        size_t row = logical_idx >> (S + S);
        size_t irow = row % Pow2_B;
        size_t icol = irow ^ (logical_idx >> M) % (Pow2_S);
        size_t phy_offset = (row << (M + S)) + (icol << M) + offset_in_bank;
        return base_ptr[phy_offset];
    }
};

template <typename T, int B, int M = 3, int S, size_t ROW, size_t COL, size_t STRIDE>
struct smem_col {
    T* base_ptr;

    static constexpr size_t Pow2_M = (1 << M);
    static constexpr size_t Pow2_S = (1 << S);
    static constexpr size_t Pow2_B = (1 << B);

    __device__ __forceinline__
    smem_col(T* ptr) : base_ptr(ptr) {}

    __device__ __forceinline__
    T& operator()(size_t logical_idx_row, size_t logical_idx_col) const {
        size_t logical_idx = logical_idx_col * STRIDE + logical_idx_row;

        // coordinate must be start of a bank
        assert(logical_idx % Pow2_M == 0);

        size_t row = logical_idx >> (Pow2_S + Pow2_S);
        size_t irow = row % Pow2_B;
        size_t icol = irow ^ (logical_idx >> Pow2_M) % (Pow2_S);
        size_t phy_offset = row << (Pow2_M + Pow2_S) + icol << Pow2_M;
        return base_ptr[phy_offset];
    }
};