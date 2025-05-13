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
#include "common.h"
namespace kernel {
enum class ElementUnaryOpType {
  EXP,
  SILU,
  GELU,
  RELU,
  CLAMP,
  SQUARE,
  SQRT,
  MULSCALAR
};

template <typename T, ElementUnaryOpType OP>
static __device__ __forceinline__ T
    perform_element_unary_op(T a, float scalar = 0.0f) {
  if constexpr (!std::is_same_v<T, __nv_bfloat16>) {
    assert(0 && "unsupport datatype in tb elementunary");
  }
  if constexpr (OP == ElementUnaryOpType::EXP) {
    return float2bfloat16(expf(bfloat162float(a)));
  } else if constexpr (OP == ElementUnaryOpType::SILU) {
    return float2bfloat16((bfloat162float(a)) *
                          (1.0f / (1.0f + expf(bfloat162float(-a)))));
  } else if constexpr (OP == ElementUnaryOpType::GELU) {
    return float2bfloat16((((float)a) / 2.0f) *
                          (1.0f + erff((bfloat162float(a)) / sqrtf(2.0f))));
  } else if constexpr (OP == ElementUnaryOpType::RELU) {
    return float2bfloat16(fmaxf(0.f, bfloat162float(a)));
  } else if constexpr (OP == ElementUnaryOpType::CLAMP) {
    return float2bfloat16(fmaxf(0.f, fminf(bfloat162float(a), 1.f)));
  } else if constexpr (OP == ElementUnaryOpType::SQUARE) {
    return float2bfloat16(bfloat162float(a) * bfloat162float(a));
  } else if constexpr (OP == ElementUnaryOpType::SQRT) {
    return float2bfloat16(sqrtf(bfloat162float(a)));
  } else if constexpr (OP == ElementUnaryOpType::MULSCALAR) {
    return float2bfloat16(scalar * bfloat162float(a));

  } else {
    assert(0 && "unsupport optype in tb elementunary");
  }

  return float2bfloat16(0.0f);
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