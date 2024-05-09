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

#include "mirage/kernel/device_tensor.h"
#include "mirage/threadblock/operator.h"
#include "mirage/threadblock/smem_tensor.h"
#include "mirage/threadblock/serializer/kernel_params.h"
#include <vector>
#include <vector_types.h>

namespace mirage {
namespace threadblock {

class ExecutionPlan {
public:
  std::vector<
      std::pair<mirage::type::TBOperatorType, std::vector<std::pair<int, int>>>>
      ops;
  // input-related fields
  std::vector<int3> input_map;
  std::vector<int> forloop_dim;
  std::vector<mirage::layout::SmemLayout> input_smem_layouts;
  // output-related fields
  int3 output_map; // assume that all output must use the same map
  int forloop_range;
  int reduction_dimx = mirage::type::TB_REDUCTION_DIMX;
  dim3 grid_dim, block_dim;
};

NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ExecutionPlan,
                                   ops,
                                   input_map,
                                   forloop_dim,
                                   input_smem_layouts,
                                   output_map,
                                   forloop_range,
                                   reduction_dimx,
                                   grid_dim,
                                   block_dim)

class Graph {
private:
  struct pair_hash {
    size_t operator()(std::pair<int, int> const &p) const;
  };

public:
  Graph(dim3 grid_dim, dim3 block_dim, int forloop_range, int reduction_dimx);
  // input operator
  STensor new_input(mirage::kernel::DTensor const &dtensor,
                    int3 input_map,
                    int forloop_dim,
                    mirage::layout::SmemLayout layout);
  TBOperator *create_input_op(mirage::kernel::DTensor const &dtensor,
                              int3 input_map,
                              int forloop_dim,
                              mirage::layout::SmemLayout layout);
  // output operator
  mirage::kernel::DTensor new_output(STensor const &stensor, int3 output_map);
  TBOperator *create_output_op(STensor const &stensor, int3 output_map);
  // matmul operator
  STensor matmul(STensor const &A, STensor const &B);
  TBOperator *create_matmul_op(STensor const &A, STensor const &B);
  // element unary operator
  STensor exp(STensor const &A);
  TBOperator *create_elementunary_op(STensor const &A,
                                     mirage::type::TBOperatorType _type);
  STensor add(STensor const &A, STensor const &B);
  STensor mul(STensor const &A, STensor const &B);
  STensor div(STensor const &A, STensor const &B);
  TBOperator *create_elementbinary_op(STensor const &A,
                                      STensor const &B,
                                      mirage::type::TBOperatorType _type);
  // reduction operator
  STensor reduction(STensor const &A, int dim);
  TBOperator *create_reduction_op(STensor const &A, int dim);

  // reduction_to_dimx operator
  STensor reduction_to_dimx(STensor const &A, int dim);
  TBOperator *create_reduction_to_dimx_op(STensor const &A, int dim);

  // concat operator
  STensor concat(STensor const &A, STensor const &B, int dim);
  TBOperator *create_concat_op(STensor const &A, STensor const &B, int dim);

  off_t allocate(STensor const &tensor);
  void free(STensor const &tensor);
  void free(std::vector<STensor> const &tensors);

  KernelParams get_kernel_params();
  NewKernelParams get_new_kernel_params(bool fingerprint) const;

  operator json() const;

public:
  dim3 grid_dim, block_dim;
  int forloop_range;
  int reduction_dimx;
  std::vector<mirage::threadblock::TBOperator *> operators;
  // memory allocator
  off_t smem_offset;
  std::vector<std::pair<off_t, size_t>> allocated_tensors;
};

} // namespace threadblock
} // namespace mirage
