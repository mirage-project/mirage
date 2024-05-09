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

#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/serializer/element_binary_serializer.h"
#include "mirage/threadblock/serializer/element_unary_serializer.h"
#include "mirage/threadblock/serializer/input_loader_serializer.h"
#include "mirage/threadblock/serializer/matmul_serializer.h"
#include "mirage/threadblock/serializer/output_saver_serializer.h"
#include "mirage/threadblock/serializer/reduction_serializer.h"
#include "mirage/threadblock/serializer/concat_serializer.h"
#include "mirage/utils/hash_utils.h"

namespace mirage {
namespace threadblock {

Graph::Graph(dim3 _grid_dim, dim3 _block_dim, int _forloop_range, int _reduction_dimx)
    : grid_dim(_grid_dim), block_dim(_block_dim),
      forloop_range(_forloop_range), reduction_dimx(_reduction_dimx),
      smem_offset(0) {
        assert(reduction_dimx > 0);
      }

size_t Graph::pair_hash::operator()(std::pair<int, int> const &p) const {
  size_t h1 = std::hash<int>{}(p.first);
  size_t h2 = std::hash<int>{}(p.second);
  hash_combine(h1, h2);
  return h1;
}

off_t Graph::allocate(STensor const &tensor) {
  off_t ret = smem_offset;

  off_t aligns_size = ((tensor.size() + 15) & ~15);
  smem_offset += aligns_size;

  assert(smem_offset <= (off_t)mirage::type::MAX_SMEM_SIZE);
  allocated_tensors.push_back(std::make_pair(ret, aligns_size));
  return ret;
}

void Graph::free(STensor const &tensor) {
  assert(allocated_tensors.size() > 0);
  assert(allocated_tensors.back().first == tensor.smem_offset);
  assert(allocated_tensors.back().second == ((tensor.size() + 15) & ~15));
  smem_offset -= allocated_tensors.back().second;
  allocated_tensors.pop_back();
}

void Graph::free(std::vector<STensor> const &tensors) {
  for (int i = tensors.size() - 1; i >= 0; i--) {
    free(tensors[i]);
  }
}

NewKernelParams Graph::get_new_kernel_params(bool fingerprint) const {
  NewKernelParams params;
  params.num_operators = operators.size();
  params.num_parameters = 0;
  params.num_dmem_inputs = 0;
  params.num_dmem_outputs = 0;

  assert(params.num_operators <= KernelParams::MAX_NUM_OPERATORS);
  // Our serializer assumes that input loaders are the first operators
  // and that output savers are the last operators
  for (size_t i = 0; i < operators.size(); i++) {
    params.operator_types[i] = operators[i]->op_type;
    switch (operators[i]->op_type) {
      case mirage::type::TB_INPUT_OP: {
        TBInputOp *input_op = static_cast<TBInputOp *>(operators[i]);
        mirage::kernel::DTensor dtensor = input_op->dtensor;
        int3 input_map = input_op->input_map;
        int forloop_dim = input_op->forloop_dim;
        if (fingerprint) {
          params.dmem_input_ptrs[params.num_dmem_inputs++] =
              input_op->dtensor.fp_ptr;
        } else {
          params.dmem_input_ptrs[params.num_dmem_inputs++] =
              input_op->dtensor.data_ptr;
        }
        // Serialize parameters for input loader
        mirage::threadblock::STensor stensor = operators[i]->output_tensors[0];
        // Assert that stensor and dtensor have the same num of dims
        int num_dims = stensor.num_dims;
        assert(num_dims == dtensor.num_dims);
        int2 dtensor_matrix_shape, stensor_matrix_shape;
        dtensor_matrix_shape = {dtensor.dim[num_dims - 2],
                                dtensor.dim[num_dims - 1]};
        stensor_matrix_shape = {stensor.dim[num_dims - 2],
                                stensor.dim[num_dims - 1]};
        int input_smem_offset = stensor.smem_offset;
        mirage::layout::DmemLayout dtensor_layout = dtensor.layout;
        mirage::layout::SmemLayout stensor_layout = stensor.layout;
        int3 input_matrix_row_offset_block_stride = {
            (input_map.x == num_dims - 2 ? stensor.dim[num_dims - 2] : 0) *
                (forloop_dim == num_dims - 2 ? this->forloop_range : 1),
            (input_map.y == num_dims - 2 ? stensor.dim[num_dims - 2] : 0) *
                (forloop_dim == num_dims - 2 ? this->forloop_range : 1),
            (input_map.z == num_dims - 2 ? stensor.dim[num_dims - 2] : 0) *
                (forloop_dim == num_dims - 2 ? this->forloop_range : 1)};
        int3 input_matrix_column_offset_block_stride = {
            (input_map.x == num_dims - 1 ? stensor.dim[num_dims - 1] : 0) *
                (forloop_dim == num_dims - 1 ? this->forloop_range : 1),
            (input_map.y == num_dims - 1 ? stensor.dim[num_dims - 1] : 0) *
                (forloop_dim == num_dims - 1 ? this->forloop_range : 1),
            (input_map.z == num_dims - 1 ? stensor.dim[num_dims - 1] : 0) *
                (forloop_dim == num_dims - 1 ? this->forloop_range : 1)};
        // int tb_offset_row = blockIdx.x * row_stride.x + blockIdx.y *
        // row_stride.y +
        //                     blockIdx.z * row_stride.z;
        // int tb_offset_column = blockIdx.x * column_stride.x +
        //                        blockIdx.y * column_stride.y +
        //                        blockIdx.z * column_stride.z;
        //  FIXME: use cutlass prologue for loading data into shared memory
        //  examples/13_two_tensor_op_fusion/threadblock/
        //  b2b_mma_pipelined_smem_accumulator.h prologue iterators
        //  input_matrix_offset_base = {tb_offset_row, tb_offset_column};
        int input_matrix_row_offset_forloop_stride = 0;
        int input_matrix_column_offset_forloop_stride = 0;
        if (forloop_dim == num_dims - 2) {
          input_matrix_row_offset_forloop_stride = stensor.dim[num_dims - 2];
        }
        if (forloop_dim == num_dims - 1) {
          input_matrix_column_offset_forloop_stride = stensor.dim[num_dims - 1];
        }
        // calculate global offset beyond the last two dimensions
        // global_offset captures offsets caused by partitioning other
        // dimensions such as batch matmul global_offset is directly added to
        // dtensor.data_ptr by the input loader
        int3 global_offset_block_stride = {0, 0, 0};
        int global_offset_forloop_stride = 0;
        if (num_dims > 2) {
          int strides[MAX_TENSOR_DIMS];
          strides[num_dims - 1] = 0;
          strides[num_dims - 2] = 0;
          strides[num_dims - 3] =
              dtensor.dim[num_dims - 2] * dtensor.dim[num_dims - 1];
          for (int j = num_dims - 4; j >= 0; j--) {
            strides[j] = strides[j + 1] * dtensor.dim[j + 1];
          }
          if (input_map.x < num_dims - 2 && input_map.x >= 0) {
            global_offset_block_stride.x = strides[input_map.x];
          }
          if (input_map.y < num_dims - 2 && input_map.y >= 0) {
            global_offset_block_stride.y = strides[input_map.y];
          }
          if (input_map.z < num_dims - 2 && input_map.z >= 0) {
            global_offset_block_stride.z = strides[input_map.z];
          }
          if (forloop_dim < num_dims - 2 && forloop_dim >= 0) {
            global_offset_forloop_stride =
                stensor.dim[forloop_dim] * strides[forloop_dim];
          }
        } // if (num_dims > 2)
        mirage::threadblock::serialize_input_loader_parameters(
            params.parameters,
            params.num_parameters,
            input_matrix_row_offset_block_stride,
            input_matrix_column_offset_block_stride,
            input_matrix_row_offset_forloop_stride,
            input_matrix_column_offset_forloop_stride,
            global_offset_block_stride,
            global_offset_forloop_stride,
            dtensor_matrix_shape,
            stensor_matrix_shape,
            dtensor_layout,
            stensor_layout,
            input_smem_offset);
        break;
      }
      case mirage::type::TB_OUTPUT_OP: {
        TBOutputOp *output_op = static_cast<TBOutputOp *>(operators[i]);
        mirage::kernel::DTensor dtensor = output_op->dtensor;
        int3 output_map = output_op->output_map;
        if (fingerprint) {
          params.dmem_output_ptrs[params.num_dmem_outputs++] =
              output_op->dtensor.fp_ptr;
        } else {
          params.dmem_output_ptrs[params.num_dmem_outputs++] =
              output_op->dtensor.data_ptr;
        }
        // Serialize parameters for input loader
        assert(operators[i]->input_tensors.size() == 1);
        assert(operators[i]->output_tensors.size() == 1);
        mirage::threadblock::STensor input_stensor =
            operators[i]->input_tensors[0];
        mirage::threadblock::STensor accum_stensor =
            operators[i]->output_tensors[0];
        // Assert that stensor and dtensor have the same num of dims
        int num_dims = input_stensor.num_dims;
        assert(num_dims == accum_stensor.num_dims);
        assert(num_dims == dtensor.num_dims);
        int2 dtensor_matrix_shape, stensor_matrix_shape;
        dtensor_matrix_shape = {dtensor.dim[num_dims - 2],
                                dtensor.dim[num_dims - 1]};
        stensor_matrix_shape = {input_stensor.dim[num_dims - 2],
                                input_stensor.dim[num_dims - 1]};
        int input_smem_offset = input_stensor.smem_offset;
        int accum_smem_offset = accum_stensor.smem_offset;
        mirage::layout::DmemLayout dtensor_layout = dtensor.layout;
        mirage::layout::SmemLayout stensor_layout = input_stensor.layout;
        int3 output_matrix_row_offset_block_stride = {
            output_map.x == num_dims - 2 ? input_stensor.dim[num_dims - 2] : 0,
            output_map.y == num_dims - 2 ? input_stensor.dim[num_dims - 2] : 0,
            output_map.z == num_dims - 2 ? input_stensor.dim[num_dims - 2] : 0};
        int3 output_matrix_column_offset_block_stride = {
            output_map.x == num_dims - 1 ? input_stensor.dim[num_dims - 1] : 0,
            output_map.y == num_dims - 1 ? input_stensor.dim[num_dims - 1] : 0,
            output_map.z == num_dims - 1 ? input_stensor.dim[num_dims - 1] : 0};
        int3 global_offset_block_stride = {0, 0, 0};
        if (num_dims > 2) {
          int strides[MAX_TENSOR_DIMS];
          strides[num_dims - 3] =
              dtensor.dim[num_dims - 2] * dtensor.dim[num_dims - 1];
          for (int j = num_dims - 4; j >= 0; j--) {
            strides[j] = strides[j + 1] * dtensor.dim[j + 1];
          }
          if (output_map.x < num_dims - 2 && output_map.x >= 0) {
            global_offset_block_stride.x = strides[output_map.x];
          }
          if (output_map.y < num_dims - 2 && output_map.y >= 0) {
            global_offset_block_stride.y = strides[output_map.y];
          }
          if (output_map.z < num_dims - 2 && output_map.z >= 0) {
            global_offset_block_stride.z = strides[output_map.z];
          }
        }
        mirage::threadblock::serialize_output_saver_parameters(
            params.parameters,
            params.num_parameters,
            output_matrix_row_offset_block_stride,
            output_matrix_column_offset_block_stride,
            global_offset_block_stride,
            dtensor_matrix_shape,
            stensor_matrix_shape,
            dtensor_layout,
            stensor_layout,
            input_smem_offset,
            accum_smem_offset);
        break;
      }
      case mirage::type::TB_MATMUL_OP: {
        assert(operators[i]->input_tensors.size() == 2);
        assert(operators[i]->output_tensors.size() == 1);
        mirage::threadblock::STensor A = operators[i]->input_tensors[0];
        mirage::threadblock::STensor B = operators[i]->input_tensors[1];
        mirage::threadblock::STensor C = operators[i]->output_tensors[0];
        int num_dims = A.num_dims;
        assert(B.num_dims == num_dims);
        assert(C.num_dims == num_dims);
        // Currently do not support batch matmul in TB
        for (int i = 0; i < num_dims - 2; i++) {
          assert(A.dim[i] == 1);
          assert(B.dim[i] == 1);
          assert(C.dim[i] == 1);
        }
        int m = A.dim[num_dims - 2];
        int n = B.dim[num_dims - 1];
        int k = A.dim[num_dims - 1];
        assert(B.dim[num_dims - 2] == k);
        assert(C.dim[num_dims - 2] == m);
        assert(C.dim[num_dims - 1] == n);
        mirage::threadblock::serialize_matmul_op_parameters(params.parameters,
                                                         params.num_parameters,
                                                         m,
                                                         n,
                                                         k,
                                                         A.smem_offset,
                                                         B.smem_offset,
                                                         C.smem_offset);
        break;
      }
      case mirage::type::TB_EXP_OP: {
        assert(operators[i]->input_tensors.size() == 1);
        assert(operators[i]->output_tensors.size() == 1);
        mirage::threadblock::STensor input = operators[i]->input_tensors[0];
        mirage::threadblock::STensor output = operators[i]->output_tensors[0];
        // assert inplace
        assert(input.smem_offset == output.smem_offset);
        assert(input.num_elements() == output.num_elements());
        mirage::threadblock::serialize_elementunary_op_parameters(
            params.parameters,
            params.num_parameters,
            input.smem_offset,
            (int)input.num_elements());
        break;
      }
      case mirage::type::TB_DIV_OP: 
      case mirage::type::TB_ADD_OP: {
        assert(operators[i]->input_tensors.size() == 2);
        assert(operators[i]->output_tensors.size() == 1);
        mirage::threadblock::STensor input1 = operators[i]->input_tensors[0];
        mirage::threadblock::STensor input2 = operators[i]->input_tensors[1];
        mirage::threadblock::STensor output = operators[i]->output_tensors[0];
        int3 input1_shape = {1, 1, 1}, input2_shape = {1, 1, 1};
        // assert that only the last three dimensions can be larger than 1
        // since we only serialize these
        for (int i = 0; i < input1.num_dims - 3; i++) {
          assert(input1.dim[i] == 1);
        }
        for (int i = 0; i < input2.num_dims - 3; i++) {
          assert(input2.dim[i] == 1);
        }
        input1_shape.z =
            input1.num_dims > 0 ? input1.dim[input1.num_dims - 1] : 1;
        input1_shape.y =
            input1.num_dims > 1 ? input1.dim[input1.num_dims - 2] : 1;
        input1_shape.x =
            input1.num_dims > 2 ? input1.dim[input1.num_dims - 3] : 1;
        input2_shape.z =
            input2.num_dims > 0 ? input2.dim[input2.num_dims - 1] : 1;
        input2_shape.y =
            input2.num_dims > 1 ? input2.dim[input2.num_dims - 2] : 1;
        input2_shape.x =
            input2.num_dims > 2 ? input2.dim[input2.num_dims - 3] : 1;
        mirage::threadblock::serialize_elementbinary_op_parameters(
            params.parameters,
            params.num_parameters,
            input1_shape,
            input2_shape,
            input1.smem_offset,
            input2.smem_offset,
            output.smem_offset);
        break;
      }
      case mirage::type::TB_REDUCTION_0_OP:
      case mirage::type::TB_REDUCTION_1_OP:
      case mirage::type::TB_REDUCTION_2_OP:
      case mirage::type::TB_REDUCTION_0_TO_DIMX_OP:
      case mirage::type::TB_REDUCTION_1_TO_DIMX_OP:
      case mirage::type::TB_REDUCTION_2_TO_DIMX_OP: {
        assert(operators[i]->input_tensors.size() == 1);
        assert(operators[i]->output_tensors.size() == 1);
        mirage::threadblock::STensor input = operators[i]->input_tensors[0];
        mirage::threadblock::STensor output = operators[i]->output_tensors[0];
        mirage::type::TBOperatorType type = operators[i]->op_type;
        int reduction_dim = -1;
        if (type >= mirage::type::TB_REDUCTION_0_TO_DIMX_OP &&
            type <= mirage::type::TB_REDUCTION_2_TO_DIMX_OP) {
          reduction_dim = type - mirage::type::TB_REDUCTION_0_TO_DIMX_OP;
        } else if (type >= mirage::type::TB_REDUCTION_0_OP &&
                   type <= mirage::type::TB_REDUCTION_2_OP) {
          reduction_dim = type - mirage::type::TB_REDUCTION_0_OP;
        } else {
          assert(false);
        }
        assert(input.num_dims == output.num_dims);
        int reduction_degree = input.num_elements() / output.num_elements();
        for (int i = 0; i < input.num_dims; i++) {
          if (i != reduction_dim) {
            assert(input.dim[i] == output.dim[i]);
          } else {
            assert(input.dim[i] == output.dim[i] * reduction_degree);
          }
        }
        int inner_range = 1;
        for (int i = reduction_dim; i < output.num_dims; i++) {
          inner_range *= output.dim[i];
        }
        mirage::threadblock::serialize_reduction_op_parameters(
            params.parameters,
            params.num_parameters,
            (int)output.num_elements(),
            reduction_degree,
            inner_range,
            input.smem_offset,
            output.smem_offset);
        break;
      }
      case mirage::type::TB_CONCAT_0_OP:
      case mirage::type::TB_CONCAT_1_OP:
      case mirage::type::TB_CONCAT_2_OP: {
        assert(operators[i]->input_tensors.size() == 2);
        assert(operators[i]->output_tensors.size() == 1);
        mirage::threadblock::STensor A = operators[i]->input_tensors[0];
        mirage::threadblock::STensor B = operators[i]->input_tensors[1];
        mirage::threadblock::STensor output = operators[i]->output_tensors[0];
        int concat_dim = operators[i]->op_type - mirage::type::TB_CONCAT_0_OP;
        assert(A.num_dims == B.num_dims);
        assert(A.num_dims == output.num_dims);
        int inner_size = 1;
        for (int i = 0; i < A.num_dims; i++) {
          if (i == concat_dim) {
            assert(A.dim[i] + B.dim[i] == output.dim[i]);
          } else {
            assert(A.dim[i] == output.dim[i]);
            assert(B.dim[i] == output.dim[i]);
          }
          if (i > concat_dim)
            inner_size = inner_size * output.dim[i];
        }
        mirage::threadblock::serialize_concat_op_parameters(
            params.parameters,
            params.num_parameters,
            (int)output.num_elements(),
            A.dim[concat_dim],
            B.dim[concat_dim],
            inner_size,
            A.smem_offset,
            B.smem_offset,
            output.smem_offset);
        break;
      }
      default: {
        assert(false && "Unsupported TB operator");
      }
    } // switch
  }   // for-loop
  // Our serializer assumes that input loaders are the first operators
  // and that output savers are the last operators
  for (int i = 0; i < params.num_dmem_inputs; i++) {
    assert(params.operator_types[i] == mirage::type::TB_INPUT_OP);
  }
  for (int i = params.num_operators - params.num_dmem_outputs;
       i < params.num_operators;
       i++) {
    assert(params.operator_types[i] == mirage::type::TB_OUTPUT_OP);
  }
  return params;
}

KernelParams Graph::get_kernel_params() {
  KernelParams params;
  params.forloop_range = this->forloop_range;
  params.num_operators = operators.size();
  params.num_smem_inputs = 0;
  params.num_smem_outputs = 0;
  params.num_dmem_inputs = 0;
  params.num_dmem_outputs = 0;

  assert(params.num_operators <= KernelParams::MAX_NUM_OPERATORS);
  for (size_t i = 0; i < operators.size(); i++) {
    params.operator_types[i] = operators[i]->op_type;
    params.operator_num_inputs[i] = operators[i]->input_tensors.size();
    params.operator_num_outputs[i] = operators[i]->output_tensors.size();
    for (int j = 0; j < params.operator_num_inputs[i]; j++) {
      params.smem_inputs[params.num_smem_inputs++] =
          operators[i]->input_tensors[j];
      assert(params.num_smem_inputs <= KernelParams::MAX_TOTAL_SMEM_INPUTS);
    }
    for (int j = 0; j < params.operator_num_outputs[i]; j++) {
      params.smem_outputs[params.num_smem_outputs++] =
          operators[i]->output_tensors[j];
      assert(params.num_smem_outputs <= KernelParams::MAX_TOTAL_SMEM_OUTPUTS);
    }
    if (operators[i]->op_type == mirage::type::TB_INPUT_OP) {
      TBInputOp *input_op = static_cast<TBInputOp *>(operators[i]);
      params.input_map[params.num_dmem_inputs] = input_op->input_map;
      params.forloop_dim[params.num_dmem_inputs] = input_op->forloop_dim;
      params.dmem_inputs[params.num_dmem_inputs++] = input_op->dtensor;
      // printf("sizeof(dtensor) = %zu\n", sizeof(input_op->dtensor));
      assert(params.num_dmem_inputs <= KernelParams::MAX_NUM_DMEM_INPUTS);
    }
    if (operators[i]->op_type == mirage::type::TB_OUTPUT_OP) {
      TBOutputOp *output_op = static_cast<TBOutputOp *>(operators[i]);
      params.output_map = output_op->output_map;
      params.dmem_outputs[params.num_dmem_outputs++] = output_op->dtensor;
      assert(params.num_dmem_outputs <= KernelParams::MAX_NUM_DMEM_OUTPUTS);
    }
  }
  return params;
}

Graph::operator json() const {
  json j = {{"graph_level", "thread_block_graph"},
            {"grid_dim", grid_dim},
            {"block_dim", block_dim},
            {"forloop_range", forloop_range},
            {"reduction_dimx", reduction_dimx},
            {"operators", {}},
            {"smem_offset", smem_offset}};
  for (TBOperator *const op : operators) {
    j["operators"].push_back(json(*op));
  }
  return j;
}

} // namespace threadblock
} // namespace mirage
