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

#ifdef DEADCODE

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
#endif

class TBConcatFingerprinter {
public:
  CUTLASS_DEVICE
  TBConcatFingerprinter(FPType *A_ptr,
                        FPType *B_ptr,
                        FPType *output_ptr,
                        int output_num_elements,
                        int A_concat_dim_size,
                        int B_concat_dim_size,
                        int inner_size,
                        int thread_id,
                        int num_threads) {
    for (int i = thread_id; i < output_num_elements; i += num_threads) {
      int inner_idx = i % inner_size;
      int concat_dim_idx = (i / inner_size) % (A_concat_dim_size + B_concat_dim_size);
      int outer_idx = (i / inner_size) / (A_concat_dim_size + B_concat_dim_size);
      if (concat_dim_idx < A_concat_dim_size) {
        int A_idx = inner_idx + concat_dim_idx * inner_size + outer_idx * inner_size * A_concat_dim_size;
        output_ptr[i] = A_ptr[A_idx];
      } else {
        int B_idx = inner_idx + (concat_dim_idx - A_concat_dim_size) * inner_size + outer_idx * inner_size * B_concat_dim_size;
        output_ptr[i] = B_ptr[B_idx];
      }
    }
  };
};

} // namespace threadblock
} // namespace mirage
