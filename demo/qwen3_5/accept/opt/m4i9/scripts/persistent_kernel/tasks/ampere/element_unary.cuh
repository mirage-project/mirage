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
#include "tasks/common/common_header.cuh"
namespace kernel {
enum class ElementUnaryOpType {
  EXP,
  SILU,
  GELU,
  RELU,
  CLAMP,
  SQUARE,
  SQRT,
  MULSCALAR,
  ADDSCALAR
};

using bfloat16 = type::bfloat16_t;

template <typename T, ElementUnaryOpType OP>
static __device__ __forceinline__ T
    perform_element_unary_op(T a, float scalar = 0.0f) {
  if constexpr (!(std::is_same_v<T, bfloat16> || std::is_same_v<T, float>)) {
    assert(0 && "unsupport datatype in tb elementunary");
  }
  if constexpr (OP == ElementUnaryOpType::EXP) {
    return bfloat16(expf(float(a)));
  } else if constexpr (OP == ElementUnaryOpType::SILU) {
    return bfloat16((float(a)) * (1.0f / (1.0f + expf(float(-a)))));
  } else if constexpr (OP == ElementUnaryOpType::GELU) {
    return bfloat16((((float)a) / 2.0f) *
                    (1.0f + erff((float(a)) / sqrtf(2.0f))));
  } else if constexpr (OP == ElementUnaryOpType::RELU) {
    return bfloat16(fmaxf(0.f, float(a)));
  } else if constexpr (OP == ElementUnaryOpType::CLAMP) {
    return bfloat16(fmaxf(0.f, fminf(float(a), 1.f)));
  } else if constexpr (OP == ElementUnaryOpType::SQUARE) {
    return bfloat16(float(a) * float(a));
  } else if constexpr (OP == ElementUnaryOpType::SQRT) {
    return bfloat16(sqrtf(float(a)));
  } else if constexpr (OP == ElementUnaryOpType::MULSCALAR) {
    return bfloat16(scalar * float(a));
  } else if constexpr (OP == ElementUnaryOpType::ADDSCALAR) {
    return bfloat16(scalar + float(a));
  } else {
    assert(0 && "unsupport optype in tb elementunary");
  }

  return bfloat16(0.0f);
}

template <typename T>
__device__ __forceinline__ T perform_element_unary_chain(T a,
                                                         float const *scalars,
                                                         int idx) {
  return a;
}

template <typename T,
          ElementUnaryOpType FirstOp,
          ElementUnaryOpType... RemainingOps>
__device__ __forceinline__ T perform_element_unary_chain(T a,
                                                         float const *scalars,
                                                         int idx) {
  T res = perform_element_unary_op<T, FirstOp>(a, scalars[idx]);
  return perform_element_unary_chain<T, RemainingOps...>(res, scalars, idx + 1);
}

// now assume input output using the same layout
template <bool ACCUM,
          typename SMEM_DST,
          typename SMEM_SRC,
          ElementUnaryOpType FirstOp,
          ElementUnaryOpType... RemainingOps>
__device__ __forceinline__ void perform_element_unary_chain_kernel(
    SMEM_DST dst, SMEM_SRC src, float const *scalars) {
#pragma unroll
  for (int elem_idx = threadIdx.x; elem_idx < SMEM_DST::size();
       elem_idx += NUM_THREADS) {
    auto value = src.at(elem_idx);
    auto result =
        perform_element_unary_chain<typename SMEM_DST::value_type,
                                    FirstOp,
                                    RemainingOps...>(value, scalars, 0);
    if (!ACCUM) {
      dst.at(elem_idx) = result;
    } else {
      dst.at(elem_idx) += result;
    }
  }
}
} // namespace kernel
