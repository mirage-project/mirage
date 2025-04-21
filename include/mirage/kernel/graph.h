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

#include "mirage/kernel/customized.h"
#include "mirage/kernel/device_tensor.h"
#include "mirage/kernel/operator.h"
#include "mirage/threadblock/graph.h"
#include <vector>

namespace mirage {
namespace kernel {

class Graph {
private:
  struct pair_hash {
    size_t operator()(std::pair<int, int> const &p) const;
  };

public:
  Graph(dim3 gpu_dim = {1, 1, 1});
  ~Graph();
  Graph(Graph const &) = delete;
  Graph &operator=(Graph const &) = delete;
  // input operator
  DTensor new_input(std::vector<int> const &dims,
                    std::vector<size_t> const &strides,
                    mirage::type::DataType data_type,
                    mirage::layout::DmemLayout layout);
  DTensor *new_input_ptr(std::vector<int> const &dims,
                         std::vector<size_t> const &strides,
                         mirage::type::DataType data_type,
                         mirage::layout::DmemLayout layout);
  KNOperator *create_input_op(std::vector<int> const &dims,
                              std::vector<size_t> const &strides,
                              mirage::type::DataType data_type,
                              mirage::layout::DmemLayout layout);
  // output operator
  void mark_output(DTensor const &A);
  void mark_output(DTensor const *A);
  void mark_output(DTensor const &A, std::vector<size_t> const &strides);
  void mark_output(DTensor const *A, std::vector<size_t> const &strides);
  KNOperator *create_output_op(DTensor const &A,
                               std::vector<size_t> const &strides);
  // matmul operator
  DTensor matmul(DTensor const &A, DTensor const &B);
  DTensor *matmul(DTensor const *A, DTensor const *B);
  KNOperator *create_matmul_op(DTensor const &A, DTensor const &B);
  // elementunary operator
  DTensor exp(DTensor const &input);
  DTensor *exp(DTensor const *input);
  DTensor square(DTensor const &input);
  DTensor *square(DTensor const *input);
  DTensor sqrt(DTensor const &input);
  DTensor *sqrt(DTensor const *input);
  DTensor silu(DTensor const &input);
  DTensor *silu(DTensor const *input);
  DTensor gelu(DTensor const &input);
  DTensor *gelu(DTensor const *input);
  DTensor relu(DTensor const &input);
  DTensor *relu(DTensor const *input);
  DTensor
      clamp(DTensor const &input, float const &min_val, float const &max_val);
  DTensor *
      clamp(DTensor const *input, float const &min_val, float const &max_val);
  DTensor elementunary(DTensor const &input,
                       mirage::type::KNOperatorType _type);
  DTensor *elementunary(DTensor const *input,
                        mirage::type::KNOperatorType _type);

  KNOperator *create_elementunary_op(DTensor const &input,
                                     mirage::type::KNOperatorType _type);

  DTensor elementunary_clamp(DTensor const &input,
                             float const &min_val,
                             float const &max_val);
  DTensor *elementunary_clamp(DTensor const *input,
                              float const &min_val,
                              float const &max_val);

  KNOperator *create_elementunary_clamp_op(DTensor const &input,
                                           float const &min_val,
                                           float const &max_val);

  // elementunary operator
  DTensor add(DTensor const &input1, DTensor const &input2);
  DTensor mul(DTensor const &input1, DTensor const &input2);
  DTensor div(DTensor const &input1, DTensor const &input2);
  DTensor pow(DTensor const &input1, DTensor const &input2);
  DTensor *add(DTensor const *input1, DTensor const *input2);
  DTensor *mul(DTensor const *input1, DTensor const *input2);
  DTensor *div(DTensor const *input1, DTensor const *input2);
  DTensor *pow(DTensor const *input1, DTensor const *input2);

  DTensor elementbinary(DTensor const &input1,
                        DTensor const &input2,
                        mirage::type::KNOperatorType _type);
  DTensor *elementbinary(DTensor const *input1,
                         DTensor const *input2,
                         mirage::type::KNOperatorType _type);
  KNOperator *create_elementbinary_op(DTensor const &input1,
                                      DTensor const &input2,
                                      mirage::type::KNOperatorType _type);
  // reduction operator
  DTensor reduction(DTensor const &input, int dim, int size = 1);
  DTensor *reduction(DTensor const *input, int dim, int size = 1);
  KNOperator *create_reduction_op(DTensor const &input, int dim, int factor);
  // normalization operator
  DTensor rms_norm(DTensor const &input,
                   std::vector<int> const &normalized_shape);
  DTensor *rms_norm(DTensor const *input,
                    std::vector<int> const &normalized_shape);
  KNOperator *create_rms_norm_op(DTensor const &input,
                                 std::vector<int> const &normalized_shape);
  DTensor rms_norm(DTensor const &input,
                   DTensor const &elementwise_afffine,
                   std::vector<int> const &normalized_shape);
  DTensor *rms_norm(DTensor const *input,
                    DTensor const *elementwise_affine,
                    std::vector<int> const &normalized_shape);
  KNOperator *create_rms_norm_op(DTensor const &input,
                                 DTensor const &elementwise_affine,
                                 std::vector<int> const &normalized_shape);
  // allreduce operator
  DTensor all_reduce(DTensor const &input, bool inplace = true);
  DTensor *all_reduce(DTensor const *input, bool inplace = true);
  KNOperator *create_all_reduce_op(DTensor const &input, bool inplace);
  // chunk operator
  std::vector<DTensor> chunk(DTensor const &input, int chunk_size, int dim);
  int chunk(DTensor const *input, int chunk_size, int dim);
  KNOperator *create_chunk_op(DTensor const &input, int chunk_size, int dim);
  // customized operator
  std::vector<DTensor> customized(std::vector<DTensor> const &inputs,
                                  mirage::threadblock::Graph const &_graph);
  int customized(std::vector<DTensor const *> inputs,
                 DTensor **outputs,
                 mirage::threadblock::Graph const *bgraph);
  KNOperator *create_customized_op(std::vector<DTensor> const &inputs,
                                   mirage::threadblock::Graph const &_graph);
  // helper functions
  int get_num_input_dtensors() const;
  int get_num_output_dtensors() const;
  int get_input_dtensors(DTensor **inputs) const;
  int get_input_dtensor_shape_and_stride(DTensor const *input,
                                         int *strides,
                                         int *dims) const;
  void generate_triton_program(char const *filepath);

  bool can_allocate(DTensor const &tensor,
                    bool allocate_fingerprint = true) const;
  bool can_allocate(size_t data_size_in_bytes, size_t fp_size_in_bytes) const;
  bool allocate(DTensor &tensor, bool allocate_fingerprint = true);
  void free(DTensor &tensor);

  // hash related functions
  size_t get_owner_independent_hash() const;

public:
  std::vector<mirage::kernel::KNOperator *> operators;
  dim3 gpu_dim;
  // memory allocator
  // device memory offset manager
  off_t dmem_data_offset, dmem_fp_offset;
  std::vector<std::pair<off_t, size_t>> allocated_data_tensors,
      allocated_fp_tensors;
  // std::unordered_map<std::pair<int, int>, DTensor, pair_hash> tensors;
  // std::unordered_map<std::pair<int, int>, std::pair<int, int>, pair_hash>
  // edges; std::vector<std::vector<SrcEdge>> edges;
  // mirage::kernel::OperatorFactory *operator_factory;

  using OpType = KNOperator;
  using TensorType = DTensor;
};

void to_json(json &j, Graph const &g);
void from_json(json const &j, Graph &g);

} // namespace kernel
} // namespace mirage
