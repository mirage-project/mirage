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

 enum class ElementUnaryOpType { EXP, SILU, GELU, RELU, CLAMP, SQUARE, SQRT };

 template <typename T, ElementUnaryOpType OP>
static __device__ __forceinline__ T
    perform_element_unary_op(T a, float scalar = 0.0f) {
  if constexpr (!(std::is_same_v<T, cutlass::half_t> ||
                  std::is_same_v<T, cutlass::bfloat16_t> ||
                  std::is_same_v<T, float> || std::is_same_v<T, __half>)) {
    assert(0 && "unsupport datatype in tb elementunary");
  }
  if constexpr (OP == ElementUnaryOpType::EXP) {
    return (T)expf((float)a);
  } else if constexpr (OP == ElementUnaryOpType::SILU) {
    return (T)(((float)a) * (1.0f / (1.0f + expf((float)-a))));
  } else if constexpr (OP == ElementUnaryOpType::GELU) {
    return (T)((((float)a) / 2.0f) * (1.0f + erff(((float)a) / sqrtf(2.0f))));
  } else if constexpr (OP == ElementUnaryOpType::RELU) {
    return (T)(fmaxf(0.f, (float)a));
  } else if constexpr (OP == ElementUnaryOpType::CLAMP) {
    return (T)(fmaxf(0.f, fminf((float)a, 1.f)));
  } else if constexpr (OP == ElementUnaryOpType::SQUARE) {
    return (T)((float)a * (float)a);
  } else if constexpr (OP == ElementUnaryOpType::SQRT) {
    return (T)(sqrtf((float)a));
  } else if constexpr (OP == ElementUnaryOpType::MULSCALAR) {
    return (T)(scalar * (float)a);
  } else {
    assert(0 && "unsupport optype in tb elementunary");
  }

  return (T)0.0;
}

// Base case: no more ops
template <typename T>
__device__ __forceinline__ T perform_element_unary_chain(T a, float scalar = 0.0f) {
  return a;
}

// Recursive case: apply first OP, then remaining OPs
template <typename T, ElementUnaryOpType FirstOp, ElementUnaryOpType... RemainingOps>
__device__ __forceinline__ T perform_element_unary_chain(T a, float scalar_first, float scalar_remaining...) {
    T first_result = perform_element_unary_op<T, FirstOp>(src[elem_idx], scalar_first);
    return perform_element_unary_chain<T, RemainingOps...>(first_result, scalar_remaining);
}

template <typename T, int NUM_THREADS, ElementUnaryOpType FirstOp, ElementUnaryOpType... RemainingOps>
__device__ __forceinline__ void perform_element_unary_chain_kernel(
    T* __restrict__ dst,
    const T* __restrict__ src,
    int size,
    int thread_idx,
    float scalar_first,
    float... scalar_remaining) {
  for (int elem_idx = thread_idx; elem_idx < size; elem_idx += NUM_THREADS) {
    T value = src[elem_idx];
    T result = perform_element_unary_chain<T, FirstOp, RemainingOps...>(value, scalar_first, scalar_remaining...);
    dst[elem_idx] = result;
  }
}