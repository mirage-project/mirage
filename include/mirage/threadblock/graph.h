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

#include "mirage/config.h"
#include "mirage/kernel/device_tensor.h"
#include "mirage/threadblock/operator.h"
#include "mirage/threadblock/serializer/kernel_params.h"
#include "mirage/threadblock/smem_tensor.h"
#include <vector>
#include <vector_types.h>

namespace mirage {
namespace threadblock {

class Graph {
private:
  struct pair_hash {
    size_t operator()(std::pair<int, int> const &p) const;
  };

public:
  Graph();
  Graph(dim3 grid_dim, dim3 block_dim, int forloop_range, int reduction_dimx);
  ~Graph();
  Graph(Graph const &) = delete;
  Graph &operator=(Graph const &) = delete;
  // input operator

  STensor new_input(mirage::kernel::DTensor const &dtensor,
                    int3 input_map,
                    int forloop_dim,
                    mirage::layout::SmemLayout layout);
  STensor *new_input(mirage::kernel::DTensor const *dtensor,
                     int3 input_map,
                     int forloop_dim,
                     mirage::layout::SmemLayout layout);
  TBOperator *create_input_op(mirage::kernel::DTensor const &dtensor,
                              int3 input_map,
                              int forloop_dim,
                              mirage::layout::SmemLayout layout);
  // output operator
  mirage::kernel::DTensor mark_output(STensor const &stensor,
                                      int3 output_map,
                                      int forloop_dim,
                                      mirage::type::TBEpilogueType epilogue);
  mirage::kernel::DTensor *new_output(STensor const *stensor,
                                      int3 output_map,
                                      int forloop_dim,
                                      mirage::type::TBEpilogueType epilogue);
  TBOperator *create_output_op(STensor const &stensor,
                               int3 output_map,
                               int forloop_dim,
                               mirage::type::TBEpilogueType epilogue);
  // matmul operator
  STensor matmul(STensor const &A, STensor const &B);
  STensor *matmul(STensor const *A, STensor const *B);
  TBOperator *create_matmul_op(STensor const &A, STensor const &B);
  // element unary operator
  STensor exp(STensor const &A);
  STensor *exp(STensor const *A);
  STensor square(STensor const &A);
  STensor *square(STensor const *A);
  STensor sqrt(STensor const &A);
  STensor *sqrt(STensor const *A);
  STensor silu(STensor const &A);
  STensor *silu(STensor const *A);
  STensor gelu(STensor const &A);
  STensor *gelu(STensor const *A);
  STensor relu(STensor const &A);
  STensor *relu(STensor const *A);
  STensor clamp(STensor const &A, float const &min_val, float const &max_val);
  STensor *clamp(STensor const *A, float const &min_val, float const &max_val);
  STensor mul_scalar(STensor const &A, float const &scalar);
  STensor *mul_scalar(STensor const *A, float const &scalar);
  STensor elementunary(STensor const &A,
                       mirage::type::TBOperatorType type,
                       float const &scalar = 0.0f);
  STensor *elementunary(STensor const *A,
                        mirage::type::TBOperatorType type,
                        float const &scalar = 0.0f);
  TBOperator *create_elementunary_op(STensor const &A,
                                     mirage::type::TBOperatorType _type,
                                     float const &scalar = 0.0f);

  STensor elementunary_clamp(STensor const &A,
                             float const &min_val,
                             float const &max_val);
  STensor *elementunary_clamp(STensor const *A,
                              float const &min_val,
                              float const &max_val);
  TBOperator *create_elementunary_clamp_op(STensor const &A,
                                           float const &min_val,
                                           float const &max_val);

  // element binary operators
  STensor add(STensor const &A, STensor const &B);
  STensor *add(STensor const *A, STensor const *B);
  STensor mul(STensor const &A, STensor const &B);
  STensor *mul(STensor const *A, STensor const *B);
  STensor div(STensor const &A, STensor const &B);
  STensor *div(STensor const *A, STensor const *B);
  STensor sub(STensor const &A, STensor const &B);
  STensor *sub(STensor const *A, STensor const *B);
  STensor pow(STensor const &A, STensor const &B);
  STensor *pow(STensor const *A, STensor const *B);

  STensor elementbinary(STensor const &A,
                        STensor const &B,
                        mirage::type::TBOperatorType type);
  STensor *elementbinary(STensor const *A,
                         STensor const *B,
                         mirage::type::TBOperatorType type);
  TBOperator *create_elementbinary_op(STensor const &A,
                                      STensor const &B,
                                      mirage::type::TBOperatorType _type);
  // reduction operator
  STensor reduction(STensor const &A, int dim);
  STensor *reduction(STensor const *A, int dim);
  TBOperator *create_reduction_op(STensor const &A, int dim);

  // reduction_to_dimx operator
  STensor reduction_to_dimx(STensor const &A, int dim);
  TBOperator *create_reduction_to_dimx_op(STensor const &A, int dim);

  // reduction_max operator
  std::vector<STensor> reduction_max(STensor const &A, int dim);
  std::vector<STensor *> reduction_max(STensor const *A, int dim);
  TBOperator *create_reduction_max_op(STensor const &A, int dim);

  // rms_norm operator
  STensor rms_norm(STensor const &A);
  STensor *rms_norm(STensor const *A);
  TBOperator *create_rms_norm_op(STensor const &A);

  // concat operator
  STensor concat(STensor const &A, STensor const &B, int dim);
  STensor *concat(STensor const *A, STensor const *B, int dim);
  TBOperator *create_concat_op(STensor const &A, STensor const &B, int dim);

  // forloop accum operator
  STensor forloop_accum(STensor const &input,
                        mirage::type::TBOperatorType type);
  STensor *forloop_accum(STensor const *input,
                         mirage::type::TBOperatorType type);
  TBOperator *create_forloop_accum_op(STensor const &input,
                                      mirage::type::TBOperatorType type);

  // forloop accum rescale operator
  STensor forloop_accum_rescale(STensor const &input,
                                STensor const &rescale,
                                mirage::type::TBOperatorType type);
  STensor *forloop_accum_rescale(STensor const *input,
                                 STensor const *rescale,
                                 mirage::type::TBOperatorType type);
  TBOperator *
      create_forloop_accum_rescale_op(STensor const &input,
                                      STensor const &rescale,
                                      mirage::type::TBOperatorType type);

  // forloop accum max operator
  STensor forloop_accum_max(STensor const &input);

  STensor *forloop_accum_max(STensor const *input);

  TBOperator *create_forloop_accum_max_op(STensor const &input);

  // fingerprint related memory management
  off_t allocate_fingerprint(STensor const &tensor);
  void free_fingerprint(STensor const &tensor);
  void free_fingerprint(std::vector<STensor> const &tensors);
  size_t calculate_shared_memory_usage(TBOperator *new_op);

  KernelParams get_kernel_params();
  NewKernelParams get_new_kernel_params(bool fingerprint) const;

  int get_smem_size_with_pipeline() const;

  operator json() const;

public:
  dim3 grid_dim, block_dim, cluster_dim;
  int forloop_range;
  int reduction_dimx;
  std::vector<mirage::threadblock::TBOperator *> operators;
  // memory allocator
  off_t smem_offset;
  std::vector<std::pair<off_t, size_t>> allocated_tensors;

  using OpType = TBOperator;
  using TensorType = STensor;
};

void from_json(json const &j, Graph &g);

} // namespace threadblock
} // namespace mirage
