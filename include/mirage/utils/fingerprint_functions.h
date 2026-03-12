/* Copyright 2023-2025 CMU
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

#include "mirage/config.h"
#include "mirage/type.h"
#include <algorithm>

namespace mirage {
namespace utils {

using namespace mirage::type;
using namespace mirage::config;
using namespace std;

#ifdef MIRAGE_FINGERPRINT_USE_CUDA
#define __execution_space__ __device__
#else
#define __execution_space__
#endif

inline __execution_space__ FPType compute_add_fingerprint(FPType a, FPType b) {
  uint32_t x = a;
  uint32_t y = b;
  return (x + y) % FP_PQ;
}

inline __execution_space__ FPType compute_mul_fingerprint(FPType a, FPType b) {
  uint32_t x = a;
  uint32_t y = b;
  uint32_t p_residual = ((x % FP_P) * (y % FP_P)) % FP_P;
  uint32_t q_residual = ((x % FP_Q) * (y % FP_Q)) % FP_Q;
  uint32_t z = p_residual * FP_Q_MUL_P_MOD_1 + q_residual * FP_P_MUL_Q_MOD_1;
  return (z % FP_PQ);
}

inline __execution_space__ FPType
    compute_div_fingerprint(FPType a,
                            FPType b,
                            FPType *div_p_lookup_table,
                            FPType *div_q_lookup_table) {
  uint32_t x = a;
  uint32_t y = b;
  uint32_t p_residual = ((x % FP_P) * div_p_lookup_table[y % FP_P]) % FP_P;
  uint32_t q_residual = ((x % FP_Q) * div_q_lookup_table[y % FP_Q]) % FP_Q;
  uint32_t z = p_residual * FP_Q_MUL_P_MOD_1 + q_residual * FP_P_MUL_Q_MOD_1;
  return (z % FP_PQ);
}

inline __execution_space__ FPType
    compute_exp_fingerprint(FPType input, FPType *exp_lookup_table) {
  FPType q_residual = input % FP_Q;
  uint32_t result = exp_lookup_table[q_residual];
  result = (result * FP_Q_MUL_P_MOD_1) % FP_PQ;
  return result;
}

inline __execution_space__ FPType compute_sqrt_fingerprint(
    FPType input, FPType *sqrt_p_lookup_table, FPType *sqrt_q_lookup_table) {
  uint32_t x = input;
  x = sqrt_p_lookup_table[x % FP_P] * FP_Q_MUL_P_MOD_1 +
      sqrt_q_lookup_table[x % FP_Q] * FP_P_MUL_Q_MOD_1;
  return x % FP_PQ;
}

inline __execution_space__ FPType compute_square_fingerprint(FPType input) {
  return compute_mul_fingerprint(input, input);
}

inline __execution_space__ FPType
    compute_silu_fingerprint(FPType input, FPType *exp_lookup_table) {
  // Note that we use $x * e^x$ as the fingerprint for SILU
  // (i.e., $x / (1+e^{-x})$, since plus one can be easier
  // implemented at any level of the GPU compute hierarchy
  // and $x / e^{-x} = x * e^x$.
  FPType q_residual = input % FP_Q;
  uint32_t result = exp_lookup_table[q_residual];
  result = (result * q_residual * FP_Q_MUL_P_MOD_1) % FP_PQ;
  return result;
}

inline __execution_space__ FPType
    compute_gelu_fingerprint(FPType input, FPType *exp_lookup_table) {
  // Approximating GeLU as x*sigmoid(1.702x)
  FPType q_residual = input % FP_Q;
  FPType scaling_factor = 2;
  FPType scaled_q_residual = (scaling_factor * q_residual) % FP_Q;
  uint32_t result = exp_lookup_table[scaled_q_residual];
  result = (result * q_residual * FP_Q_MUL_P_MOD_1) % FP_PQ;
  return result;
}

inline __execution_space__ FPType compute_clamp_fingerprint(FPType input) {
  // We use min(max(FP_Q/3, input), FP_Q*2/3) to approximate clamp
  // Note that we ignore the input arguments to clamp
  // https://pytorch.org/docs/main/generated/torch.clamp.html
  uint32_t q_residual = input % FP_Q;
  uint32_t p_residual = input % FP_P;
  q_residual = min(2 * FP_Q / 3, max(FP_Q / 3, (int)q_residual));
  p_residual = min(2 * FP_P / 3, max(FP_P / 3, (int)p_residual));
  uint32_t z = p_residual * FP_Q_MUL_P_MOD_1 + q_residual * FP_P_MUL_Q_MOD_1;
  return z % FP_PQ;
}

inline __execution_space__ FPType compute_relu_fingerprint(FPType input) {
  // We use max(FP_Q/2, input) to approximate relu
  uint32_t q_residual = input % FP_Q;
  uint32_t p_residual = input % FP_P;
  q_residual = max(FP_Q / 2, (int)q_residual);
  p_residual = max(FP_P / 2, (int)p_residual);
  uint32_t z = p_residual * FP_Q_MUL_P_MOD_1 + q_residual * FP_P_MUL_Q_MOD_1;
  return z % FP_PQ;
}

inline __execution_space__ FPType compute_pow_fingerprint(FPType base,
                                                          FPType exponent) {
  uint32_t base_p = base % FP_P;
  uint32_t base_q = base % FP_Q;
  uint32_t exp = (uint32_t)exponent;

  uint32_t result_p = 1;
  uint32_t result_q = 1;

  while (exp > 0) {
    if (exp & 1) {
      result_p = (result_p * base_p) % FP_P;
      result_q = (result_q * base_q) % FP_Q;
    }
    base_p = (base_p * base_p) % FP_P;
    base_q = (base_q * base_q) % FP_Q;
    exp >>= 1;
  }

  uint32_t z =
      (result_p * FP_Q_MUL_P_MOD_1 + result_q * FP_P_MUL_Q_MOD_1) % FP_PQ;
  return z;
}

inline __execution_space__ void accum_fingerprint(FPType &accum, FPType x) {
  accum = compute_add_fingerprint(accum, x);
}

inline __execution_space__ void accum_square_fingerprint(FPType &accum,
                                                         FPType x) {
  FPType square = compute_square_fingerprint(x);
  accum = compute_add_fingerprint(accum, square);
}

#undef __execution_space__

#ifdef MIRAGE_FINGERPRINT_USE_CPU
inline void compute_matmul_fingerprint(
    FPType *A_ptr, FPType *B_ptr, FPType *C_ptr, int B, int M, int N, int K) {
  memset(C_ptr, 0, B * M * N * sizeof(FPType));
  for (int b = 0; b < B; ++b) {
    for (int i = 0; i < M; ++i) {
      for (int k = 0; k < K; ++k) {
        for (int j = 0; j < N; ++j) {
          accum_fingerprint(
              C_ptr[b * M * N + i * N + j],
              compute_mul_fingerprint(A_ptr[b * M * K + i * K + k],
                                      B_ptr[b * K * N + k * N + j]));
        }
      }
    }
  }
}

inline void compute_rms_norm_fingerprint(FPType *input_ptr,
                                         FPType *output_ptr,
                                         FPType *div_p_lookup_table,
                                         FPType *div_q_lookup_table,
                                         FPType *sqrt_p_lookup_table,
                                         FPType *sqrt_q_lookup_table,
                                         int num_samples,
                                         int norm_size) {
  for (int i = 0; i < num_samples; ++i) {
    FPType square_sum = 0;
    for (int k = 0; k < norm_size; k++) {
      FPType x = input_ptr[i * norm_size + k];
      accum_square_fingerprint(square_sum, x);
    }
    // Compute rooted mean square
    FPType rms = 0;
    {
      FPType x = square_sum;
      FPType n = norm_size % FP_PQ;
      // Compute z = x / n
      FPType z =
          compute_div_fingerprint(x, n, div_p_lookup_table, div_q_lookup_table);
      // Perform sqrt for root-mean-square
      rms =
          compute_sqrt_fingerprint(z, sqrt_p_lookup_table, sqrt_q_lookup_table);
    }
    for (int k = 0; k < norm_size; k++) {
      FPType x = input_ptr[i * norm_size + k];
      // Compute x / rms
      FPType z = compute_div_fingerprint(
          x, rms, div_p_lookup_table, div_q_lookup_table);
      output_ptr[i * norm_size + k] = z;
    }
  }
}
#endif

} // namespace utils
} // namespace mirage
