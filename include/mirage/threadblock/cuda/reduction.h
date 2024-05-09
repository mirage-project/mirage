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

#include "mirage/utils/cuda_helper.h"
#include "mirage/utils/static_switch.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"

namespace mirage {
namespace threadblock {

using namespace cutlass;
using namespace mirage::type;

template <typename ElementType>
class RedunctionExecutor {
public:
  // reference implementation: ReduceSameRow function from
  // cutlass/examples/41_fused_multi_head_attention/gemm/mma_accum_lambda_iterator.h
  CUTLASS_DEVICE
  RedunctionExecutor(mirage::type::TBOperatorType type,
                     char *smem_buffer,
                     STensor const &input,
                     STensor const &output,
                     int thread_id,
                     int num_threads) {
    int reduction_dim = mirage::utils::get_reduction_dim(type);
    int num_dims = output.num_dims;
    ElementType *input_ptr = (ElementType *)(smem_buffer + input.smem_offset);
    ElementType *output_ptr = (ElementType *)(smem_buffer + output.smem_offset);

    int num_elements = output.num_elements();
    int output_columns = output.dim[num_dims - 1];
    int input_columns = input.dim[num_dims - 1];
    if (reduction_dim == num_dims - 2) {
      // Reduce along the row dim
      int output_rows = output.dim[num_dims - 2];
      int kK = input.dim[num_dims - 2] / output.dim[num_dims - 2];
      for (int i = 0; i < num_elements; i += 1) {
        int no = i / output_columns;
        int m = i % output_columns;
        float sum = 0.0f;
        for (int j = threadIdx.x; j < kK; j += num_threads) {
          int ni = no + j * output_rows;
          sum = static_cast<float>(input_ptr[ni * input_columns + m]);
          block_sum_fp32(sum);
        }
        output_ptr[i] = static_cast<ElementType>(sum);
      }
    } else if (reduction_dim == num_dims - 1) {
      int kK = input.dim[num_dims - 1] / output.dim[num_dims - 1];
      for (int i = 0; i < num_elements; i += 1) {
        int n = i / output_columns;
        float sum = 0.0f;
        for (int j = threadIdx.x; j < kK; j += num_threads) {
          int m = (i % output_columns) + j * output_columns;
          sum = static_cast<float>(input_ptr[n * input_columns + m]);
          block_sum_fp32(static_cast<float>(sum));
        }
        output_ptr[i] = static_cast<ElementType>(sum);
      }
    } else {
      assert(false && "Unimplemented");
    }
  }
};

template <typename ElementType>
CUTLASS_DEVICE void perform_reduction(ElementType *input_ptr,
                                      ElementType *output_ptr,
                                      int num_input_elements,
                                      int num_output_elements,
                                      int num_threads) {
  // FIXME: note that this implementation is an approximate
  // it access the same as of input/output stensors but does not
  // correctly calculate the final result
  ElementType sum = static_cast<ElementType>(0.0f);
  for (int i = threadIdx.x; i < num_input_elements; i += num_threads) {
    sum += input_ptr[i];
  }
  for (int i = threadIdx.x; i < num_output_elements; i += num_threads) {
    output_ptr[i] = sum;
  }
}

template <typename ElementType>
class SimpleRedunctionExecutor {
public:
  CUTLASS_DEVICE
  SimpleRedunctionExecutor(//mirage::type::TBOperatorType type,
                           ElementType *input_ptr,
                           ElementType *output_ptr,
                           int output_num_elements,
                           int reduction_degree,
                           int inner_range,
                           int thread_id,
                           int num_threads) {
    // int reduction_dim = mirage::utils::get_reduction_dim(type);
    // int num_dims = output.num_dims;
    //ElementType *input_ptr = (ElementType *)(smem_buffer + input.smem_offset);
    //ElementType *output_ptr = (ElementType *)(smem_buffer + output.smem_offset);

    //int num_output_elements = output.num_elements();
    //int num_input_elements = input.num_elements();
    // int reduction_degree = num_input_elements / num_output_elements;
    perform_reduction<ElementType>(input_ptr,
                                   output_ptr,
                                   output_num_elements * reduction_degree,
                                   output_num_elements,
                                   num_threads);
  }
};

class TBReductionFingerprinter {
public:
  CUTLASS_DEVICE
  TBReductionFingerprinter(mirage::type::TBOperatorType type,
                           FPType *input_ptr,
                           FPType *output_ptr,
                           int output_num_elements,
                           int reduction_degree,
                           int inner_range,
                           int thread_id,
                           int num_threads) {
    // for a reduction_dim, inner_range is calculated as follows
    // inner_range = 1;
    // for (int i = reduction_dim; i < num_dims; i++)
    //   inner_range *= output.dim[i]
    // Note that the reduction_dim size itself is included in inner_range

    // input position = (i / inner_range) * (inner_range * reduction_degree)
    // + i % inner_range + k * inner_range
    for (int i = thread_id; i < output_num_elements; i += num_threads) {
      int pos = (i / inner_range) * (inner_range * reduction_degree)
              + i % inner_range;
      uint32_t result = 0;
      for (int k = 0; k < reduction_degree; k++) {
        result = (result + input_ptr[pos]) % FP_PQ;
        pos += inner_range;
      }
      output_ptr[i] = result;
    }
#ifdef DEADCODE
    int reduction_dim = mirage::utils::get_reduction_dim(type);
    int num_dims = output.num_dims;
    FPType *input_ptr = (FPType *)(smem_buffer + input.smem_offset);
    FPType *output_ptr = (FPType *)(smem_buffer + output.smem_offset);
    int num_elements = output.num_elements();
    int output_columns = output.dim[num_dims - 1];
    int input_columns = input.dim[num_dims - 1];
    if (reduction_dim == num_dims - 2) {
      // Reduce along the row dim
      int output_rows = output.dim[num_dims - 2];
      int kK = input.dim[num_dims - 2] / output.dim[num_dims - 2];
      for (int i = thread_id; i < num_elements; i += num_threads) {
        uint32_t result = 0;
        int no = i / output_columns;
        int m = i % output_columns;
        for (int k = 0; k < kK; k++) {
          int ni = no + k * output_rows;
          result = (result + input_ptr[ni * input_columns + m]) % FP_PQ;
        }
        output_ptr[i] = result;
      }
    } else if (reduction_dim == num_dims - 1) {
      // Reduce along the column dim
      int kK = input.dim[num_dims - 1] / output.dim[num_dims - 1];
      for (int i = thread_id; i < num_elements; i += num_threads) {
        uint32_t result = 0;
        int n = i / output_columns;
        int mo = i % output_columns;
        for (int k = 0; k < kK; k++) {
          int mi = mo + k * output_columns;
          result = (result + input_ptr[n * input_columns + mi]) % FP_PQ;
        }
        output_ptr[i] = result;
      }
    } else {
      assert(false && "Unimplemented");
    }
#endif
  };
};

} // namespace threadblock
} // namespace mirage
