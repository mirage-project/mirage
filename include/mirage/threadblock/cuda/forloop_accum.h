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
#include "mirage/utils/fingerprint_functions.h"

namespace mirage {
namespace threadblock {

using namespace cutlass;
using namespace mirage::type;
using namespace mirage::config;
using namespace mirage::utils;

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
        FPType old_output = reset_output ? 0 : output_ptr[idx];
        output_ptr[idx] = compute_add_fingerprint(old_output, input_ptr[idx]);
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
        FPType result = 0;
        for (int k = 0; k < per_iter_reduction_degree; k++) {
          FPType x = input_ptr[pos];
          if (compute_square) {
            x = compute_mul_fingerprint(x, x);
          }
          result = compute_add_fingerprint(result, x);
          pos += inner_range;
        }
        FPType old_output = reset_output ? 0 : output_ptr[idx];
        output_ptr[idx] = compute_add_fingerprint(old_output, result);
        if (post_process && type == TB_FORLOOP_ACCUM_RED_LD_MEAN_OP) {
          // Perform post processing for mean reduction
          // num of elements = num_forloop_iters * per_iter_reduction_degree
          FPType x = output_ptr[idx];
          FPType n = (num_forloop_iters * per_iter_reduction_degree) % FP_PQ;
          // Compute z = x / n
          FPType z = compute_div_fingerprint(
              x, n, div_p_lookup_table, div_q_lookup_table);
          output_ptr[idx] = z;
        }
        if (post_process && type == TB_FORLOOP_ACCUM_RED_LD_RMS_OP) {
          // Perform post processing for RMS reduction
          // num of elements = num_forloop_iters * per_iter_reduction_degree
          FPType x = output_ptr[idx];
          FPType n = (num_forloop_iters * per_iter_reduction_degree) % FP_PQ;
          // Compute z = x / n
          FPType z = compute_div_fingerprint(
              x, n, div_p_lookup_table, div_q_lookup_table);
          // Perform sqrt for root-mean-square
          z = compute_sqrt_fingerprint(
              z, sqrt_p_lookup_table, sqrt_q_lookup_table);
          output_ptr[idx] = z;
        }
      }
    } else {
      assert(false && "Unsupported accum type");
    }
  }
};

} // namespace threadblock
} // namespace mirage
