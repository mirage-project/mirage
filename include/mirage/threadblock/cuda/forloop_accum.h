/* Copyright 2023-2024 CMU
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

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"

namespace mirage {
namespace threadblock {

using namespace cutlass;
using namespace mirage::type;
using namespace mirage::config;

class TBForloopAccumFingerprinter {
public:
  CUTLASS_DEVICE
  TBForloopAccumFingerprinter(TBOperatorType type,
                              FPType *input_ptr,
                              FPType *output_ptr,
                              FPType *div_p_lookup_table,
                              FPType *div_q_lookup_table,
                              FPType *sqrt_p_lookup_table,
                              FPType *sqrt_q_lookup_table,
                              int output_num_elements,
                              int per_iter_reduction_degree,
                              int inner_range,
                              int num_forloop_iters,
                              bool reset_output,
                              bool post_process,
                              int thread_id,
                              int num_threads) {
    // for reduction accumulation such as SUM, MEAN, RMS, STD
    // for a reduction_dim, inner_range is calculated as follows
    // inner_range = 1;
    // for (int i = reduction_dim; i < num_dims; i++)
    //   inner_range *= output.dim[i]
    // Note that the reduction_dim size itself is included in inner_range

    // input position = (i / inner_range) * (inner_range * reduction_degree)
    // + i % inner_range + k * inner_range

    // For non-reduction accumulation: inner_range = 1, reduction_degree = 1
    if (type == mirage::type::TB_FORLOOP_ACCUM_NO_RED_OP) {
      for (int idx = thread_id; idx < output_num_elements; idx += num_threads) {
        uint32_t old_output = reset_output ? 0 : output_ptr[idx];
        output_ptr[idx] = (old_output + input_ptr[idx]) % FP_PQ;
      }
    } else if (type == TB_FORLOOP_ACCUM_RED_LD_SUM_OP ||
               type == TB_FORLOOP_ACCUM_RED_LD_MEAN_OP ||
               type == TB_FORLOOP_ACCUM_RED_LD_RMS_OP ||
               type == TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP) {
      bool compute_square = false;
      if (type == TB_FORLOOP_ACCUM_RED_LD_RMS_OP) {
        compute_square = true;
      }
      for (int idx = thread_id; idx < output_num_elements; idx += num_threads) {
        int pos =
            (idx / inner_range) * (inner_range * per_iter_reduction_degree) +
            idx % inner_range;
        uint32_t result = 0;
        for (int k = 0; k < per_iter_reduction_degree; k++) {
          uint32_t x = input_ptr[pos];
          if (compute_square) {
            x = (x * x) % FP_PQ;
          }
          result = (result + x) % FP_PQ;
          pos += inner_range;
        }
        uint32_t old_output = reset_output ? 0 : output_ptr[idx];
        output_ptr[idx] = (old_output + result) % FP_PQ;
        if (post_process && type == TB_FORLOOP_ACCUM_RED_LD_MEAN_OP) {
          // Perform post processing for mean reduction
          // num of elements = num_forloop_iters * per_iter_reduction_degree
          uint32_t x = output_ptr[idx];
          uint32_t n = num_forloop_iters * per_iter_reduction_degree;
          // Compute z = x / n
          uint32_t z =
              (x % FP_P) * div_p_lookup_table[n % FP_P] * FP_Q_MUL_P_MOD_1 +
              (x % FP_Q) * div_q_lookup_table[n % FP_Q] * FP_P_MUL_Q_MOD_1;
          output_ptr[idx] = z % FP_PQ;
        }
        if (post_process && type == TB_FORLOOP_ACCUM_RED_LD_RMS_OP) {
          // Perform post processing for RMS reduction
          // num of elements = num_forloop_iters * per_iter_reduction_degree
          uint32_t x = output_ptr[idx];
          uint32_t n = num_forloop_iters * per_iter_reduction_degree;
          // Compute z = x / n
          uint32_t z =
              (x % FP_P) * div_p_lookup_table[n % FP_P] * FP_Q_MUL_P_MOD_1 +
              (x % FP_Q) * div_q_lookup_table[n % FP_Q] * FP_P_MUL_Q_MOD_1;
          // Perform sqrt for root-mean-square
          z = sqrt_p_lookup_table[z % FP_P] * FP_Q_MUL_P_MOD_1 +
              sqrt_q_lookup_table[z % FP_Q] * FP_P_MUL_Q_MOD_1;
          output_ptr[idx] = z % FP_PQ;
        }
      }
    } else {
      assert(false && "Unsupported accum type");
    }
  }
};

} // namespace threadblock
} // namespace mirage
