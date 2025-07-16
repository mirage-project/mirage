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

#include "mirage/kernel/customized.h"
#include "mirage/kernel/device_memory_manager.h"
#include "mirage/kernel/graph.h"
#include "mirage/threadblock/element_unary.h"
#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/operator.h"
#include "mirage/threadblock/reduction.h"
#include "mirage/threadblock/smem_tensor.h"
#include "mirage/utils/fingerprint_functions.h"
#include "mirage/utils/hash_utils.h"
#include "omp.h"
#include <cassert>

namespace mirage {
namespace kernel {

using mirage::threadblock::STensor;

std::vector<DTensor> Graph::customized(std::vector<DTensor> const &inputs,
                                       threadblock::Graph const &bgraph) {
  KNOperator *op = create_customized_op(inputs, bgraph);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors;
}

int Graph::customized(std::vector<DTensor const *> _inputs,
                      DTensor **outputs,
                      mirage::threadblock::Graph const *bgraph) {
  std::vector<DTensor> inputs;
  for (auto const &t : _inputs) {
    inputs.push_back(t == nullptr ? DTensor::EMPTY_TENSOR : *t);
  }
  KNOperator *op = create_customized_op(inputs, *bgraph);
  assert(op != nullptr);
  operators.push_back(op);
  for (size_t i = 0; i < op->output_tensors.size(); i++) {
    outputs[i] = &op->output_tensors[i];
  }
  return op->output_tensors.size();
}

KNOperator *Graph::create_customized_op(std::vector<DTensor> const &inputs,
                                        threadblock::Graph const &_graph) {
  // Assert that _graph's dtensor inputs align with inputs
  {
    int num_inputs = 0;
    for (auto const &op : _graph.operators) {
      if (op->op_type == mirage::type::TB_INPUT_OP) {
        mirage::threadblock::TBInputOp const *input_op =
            static_cast<mirage::threadblock::TBInputOp const *>(op);
        assert(inputs[num_inputs] == input_op->dtensor);
        num_inputs++;
      }
    }
    assert(num_inputs == (int)inputs.size());
  }
  // Calculate fingerprint sizes
  size_t output_data_size = 0, output_fp_size = 0;
  for (threadblock::TBOperator *op : _graph.operators) {
    if (op->op_type == type::TBOperatorType::TB_OUTPUT_OP) {
      output_data_size +=
          static_cast<threadblock::TBOutputOp *>(op)->dtensor.data_size();
      output_fp_size += static_cast<threadblock::TBOutputOp *>(op)
                            ->dtensor.fingerprint_size();
    }
  }

  if (!can_allocate(output_data_size, output_fp_size)) {
    return nullptr;
  }

  KNCustomizedOp *op = new KNCustomizedOp(this, inputs, _graph);
  return op;
}

KNCustomizedOp::KNCustomizedOp(mirage::kernel::Graph *_kgraph,
                               std::vector<DTensor> const &_inputs,
                               mirage::threadblock::Graph const &_graph)
    : KNOperator(_kgraph, mirage::type::KN_CUSTOMIZED_OP, _inputs),
      bgraph(_graph.grid_dim,
             _graph.block_dim,
             _graph.forloop_range,
             _graph.reduction_dimx) {
  size_t input_idx = 0;
  for (auto const &op : _graph.operators) {
    std::vector<STensor> my_inputs;
    std::vector<std::pair<int, int>> indices;
    for (size_t i = 0; i < op->input_tensors.size(); i++) {
      int op_idx = -1, ts_idx = op->input_tensors[i].owner_ts_idx;
      for (size_t l = 0; l < _graph.operators.size(); l++) {
        if (_graph.operators[l] == op->input_tensors[i].owner_op) {
          assert(op_idx == -1);
          op_idx = static_cast<int>(l);
        }
      }
      assert(op_idx != -1);
      my_inputs.push_back(bgraph.operators[op_idx]->output_tensors[ts_idx]);
      indices.push_back({op_idx, ts_idx});
    }
    switch (op->op_type) {
      case mirage::type::TB_INPUT_OP: {
        assert(my_inputs.size() == 0);
        mirage::threadblock::TBInputOp *input_op =
            static_cast<mirage::threadblock::TBInputOp *>(op);
        DTensor const &dtensor = _inputs[input_idx++];
        bgraph.new_input(dtensor,
                         input_op->input_map,
                         input_op->forloop_dim,
                         input_op->output_tensors[0].layout,
                         input_op->output_tensors[0].store_in_dmem);
        break;
      }
      case mirage::type::TB_OUTPUT_OP: {
        assert(my_inputs.size() == 1);
        mirage::threadblock::TBOutputOp *output_op =
            static_cast<mirage::threadblock::TBOutputOp *>(op);
        DTensor dtensor = bgraph.mark_output(my_inputs[0],
                                             output_op->output_map,
                                             output_op->forloop_dim,
                                             output_op->epilogue);
        dtensor.owner_op = this;
        dtensor.owner_ts_idx = static_cast<int>(output_tensors.size());
        dtensor.guid = DTensor::next_guid++;
        // DeviceMemoryManager *dmm = DeviceMemoryManager::get_instance();
        // dmm->allocate(dtensor);
        kgraph->allocate(dtensor);
        // Update dtensor saved by the output operator
        {
          assert(bgraph.operators.back()->op_type ==
                 mirage::type::TB_OUTPUT_OP);
          mirage::threadblock::TBOutputOp *output =
              static_cast<mirage::threadblock::TBOutputOp *>(
                  bgraph.operators.back());
          output->dtensor = dtensor;
        }
        output_tensors.push_back(dtensor);
        break;
      }
      case mirage::type::TB_MATMUL_OP: {
        assert(my_inputs.size() == 2);
        bgraph.matmul(my_inputs[0], my_inputs[1]);
        break;
      }
      case mirage::type::TB_EXP_OP:
      case mirage::type::TB_SQUARE_OP:
      case mirage::type::TB_SQRT_OP:
      case mirage::type::TB_SILU_OP:
      case mirage::type::TB_GELU_OP:
      case mirage::type::TB_RELU_OP:
      case mirage::type::TB_CLAMP_OP:
      case mirage::type::TB_MUL_SCALAR_OP: {
        assert(my_inputs.size() == 1);
        mirage::threadblock::TBElementUnaryOp const *cur_op =
            dynamic_cast<mirage::threadblock::TBElementUnaryOp const *>(op);
        bgraph.elementunary(my_inputs[0], cur_op->op_type, cur_op->scalar);
        break;
      }
      case mirage::type::TB_ADD_OP:
      case mirage::type::TB_MUL_OP:
      case mirage::type::TB_DIV_OP:
      case mirage::type::TB_SUB_OP:
      case mirage::type::TB_POW_OP: {
        assert(my_inputs.size() == 2);
        bgraph.elementbinary(my_inputs[0], my_inputs[1], op->op_type);
        break;
      }
      case mirage::type::TB_REDUCTION_0_OP:
      case mirage::type::TB_REDUCTION_1_OP:
      case mirage::type::TB_REDUCTION_2_OP: {
        assert(my_inputs.size() == 1);
        int reduce_dim = op->op_type - mirage::type::TB_REDUCTION_0_OP;
        bgraph.reduction(my_inputs[0], reduce_dim);
        break;
      }
      case mirage::type::TB_REDUCTION_0_TO_DIMX_OP:
      case mirage::type::TB_REDUCTION_1_TO_DIMX_OP:
      case mirage::type::TB_REDUCTION_2_TO_DIMX_OP: {
        assert(my_inputs.size() == 1);
        int reduce_dim = op->op_type - mirage::type::TB_REDUCTION_0_TO_DIMX_OP;
        bgraph.reduction_to_dimx(my_inputs[0], reduce_dim);
        break;
      }
      case mirage::type::TB_REDUCTION_0_MAX_OP:
      case mirage::type::TB_REDUCTION_1_MAX_OP:
      case mirage::type::TB_REDUCTION_2_MAX_OP: {
        assert(my_inputs.size() == 1);
        int reduce_dim = op->op_type - mirage::type::TB_REDUCTION_0_MAX_OP;
        bgraph.reduction_max(my_inputs[0], reduce_dim);
        break;
      }
      case mirage::type::TB_RMS_NORM_OP: {
        assert(my_inputs.size() == 1);
        bgraph.rms_norm(my_inputs[0]);
        break;
      }
      case mirage::type::TB_CONCAT_0_OP:
      case mirage::type::TB_CONCAT_1_OP:
      case mirage::type::TB_CONCAT_2_OP: {
        assert(my_inputs.size() == 2);
        int concat_dim = op->op_type - mirage::type::TB_CONCAT_FIRST_OP_ID;
        bgraph.concat(my_inputs[0], my_inputs[1], concat_dim);
        break;
      }
      case mirage::type::TB_FORLOOP_ACCUM_NO_RED_OP:
      case mirage::type::TB_FORLOOP_ACCUM_RED_LD_SUM_OP:
      case mirage::type::TB_FORLOOP_ACCUM_RED_LD_MEAN_OP:
      case mirage::type::TB_FORLOOP_ACCUM_RED_LD_RMS_OP:
      case mirage::type::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP: {
        assert(my_inputs.size() == 1);
        bgraph.forloop_accum(my_inputs[0], op->op_type);
        break;
      }
      case mirage::type::TB_FORLOOP_ACCUM_NO_RED_RESCALE_OP:
      case mirage::type::TB_FORLOOP_ACCUM_RED_LD_SUM_RESCALE_OP: {
        assert(my_inputs.size() == 2);
        bgraph.forloop_accum_rescale(my_inputs[0], my_inputs[1], op->op_type);
        break;
      }
      case mirage::type::TB_FORLOOP_ACCUM_MAX_OP: {
        assert(my_inputs.size() == 1);
        bgraph.forloop_accum_max(my_inputs[0]);
        break;
      }
      default: {
        assert(false && "Unsupported threadblock operator");
      }
    }
  }
}

void KNCustomizedOp::get_bgraph(mirage::threadblock::Graph **bgraph_) {
  *bgraph_ = &(this->bgraph);
}

KNCustomizedOp::~KNCustomizedOp() {
  for (int i = output_tensors.size() - 1; i >= 0; i--) {
    kgraph->free(output_tensors[i]);
  }
}

KNCustomizedOp::operator json() const {
  return json{{"op_type", op_type},
              {"input_tensors", input_tensors},
              {"output_tensors", output_tensors},
              {"bgraph", bgraph}};
}

size_t KNCustomizedOp::get_owner_independent_hash() const {
  assert(false && "To be implemented");
}

#ifdef MIRAGE_FINGERPRINT_USE_CPU

bool KNCustomizedOp::fingerprint(void) {
  using threadblock::STensor;
  using threadblock::TBOperator;
  using type::FPType;

  kernel::DeviceMemoryManager *dmm =
      kernel::DeviceMemoryManager::get_instance();

  auto nd_idx_to_linear =
      [](std::vector<size_t> const &indices, int num_dims, int const *dim) {
        size_t linear_idx = 0;
        for (int d = 0; d < num_dims; ++d) {
          assert(indices[d] < dim[d]);
          linear_idx = linear_idx * dim[d] + indices[d];
        }
        return linear_idx;
      };

  auto linear_idx_to_nd = [](size_t idx, int num_dims, int const *dim) {
    std::vector<size_t> indices(num_dims, 0);
    for (int d = num_dims - 1; d >= 0; --d) {
      indices[d] = idx % dim[d];
      idx /= dim[d];
    }
    return indices;
  };

  auto compute_operator = [&](TBOperator const *op,
                              char *smem_buffer,
                              bool is_first_iteration,
                              bool is_last_iteration) {
    assert(op->op_type != mirage::type::TB_INPUT_OP &&
           op->op_type != mirage::type::TB_OUTPUT_OP);
    // Operators after accumulation are only performed on the last iteration
    if (!is_last_iteration && op->input_tensors[0].after_accum) {
      return;
    }
    // Prepare input and output buffers
    std::vector<FPType *> input_buffers, output_buffers;
    for (STensor const &input : op->input_tensors) {
      input_buffers.push_back(
          reinterpret_cast<FPType *>(smem_buffer + input.smem_offset));
    }
    for (STensor const &output : op->output_tensors) {
      output_buffers.push_back(
          reinterpret_cast<FPType *>(smem_buffer + output.smem_offset));
    }
    // Perform the operation
    switch (op->op_type) {
      case type::TB_MATMUL_OP: {
        int M = op->input_tensors[0].dim[0];
        int N = op->input_tensors[1].dim[1];
        int K = op->input_tensors[0].dim[1];
        utils::compute_matmul_fingerprint(
            input_buffers[0], input_buffers[1], output_buffers[0], 1, M, N, K);
        break;
      }
      case type::TB_DIV_OP: {
        for (int i = 0; i < op->output_tensors[0].num_elements(); ++i) {
          int input1_stride = 1, input1_idx = 0;
          int input2_stride = 1, input2_idx = 0;
          {
            int idx = i;
            for (int d = op->input_tensors[0].num_dims - 1; d >= 0; --d) {
              input1_idx += (idx % op->input_tensors[0].dim[d]) * input1_stride;
              input1_stride *= op->input_tensors[0].dim[d];
              input2_idx += (idx % op->input_tensors[1].dim[d]) * input2_stride;
              input2_stride *= op->input_tensors[1].dim[d];
              idx /= op->input_tensors[0].dim[d];
            }
          }
          output_buffers[0][i] =
              utils::compute_div_fingerprint(input_buffers[0][input1_idx],
                                             input_buffers[1][input2_idx],
                                             dmm->div_p_lookup_table,
                                             dmm->div_q_lookup_table);
        }
        break;
      }
      case type::TB_FORLOOP_ACCUM_NO_RED_OP: {
        if (is_first_iteration) {
          memset(output_buffers[0],
                 0,
                 sizeof(FPType) * op->output_tensors[0].num_elements());
        }
        for (size_t i = 0; i < op->output_tensors[0].num_elements(); ++i) {
          output_buffers[0][i] = utils::compute_add_fingerprint(
              output_buffers[0][i], input_buffers[0][i]);
        }
        break;
      }
      case type::TB_FORLOOP_ACCUM_RED_LD_RMS_OP: {
        if (is_first_iteration) {
          memset(output_buffers[0],
                 0,
                 sizeof(FPType) * op->output_tensors[0].num_elements());
        }
        size_t reduction_degree =
            op->input_tensors[0].dim[op->input_tensors[0].num_dims - 1];
        for (size_t i = 0; i < op->output_tensors[0].num_elements(); ++i) {
          FPType partial_accum_result = 0;
          for (size_t j = 0; j < reduction_degree; ++j) {
            utils::accum_square_fingerprint(
                partial_accum_result,
                input_buffers[0][i * reduction_degree + j]);
          }
          utils::accum_fingerprint(output_buffers[0][i], partial_accum_result);
        }
        if (is_last_iteration) {
          FPType n = reduction_degree * bgraph.forloop_range % config::FP_PQ;
          for (size_t i = 0; i < op->output_tensors[0].num_elements(); ++i) {
            output_buffers[0][i] =
                utils::compute_div_fingerprint(output_buffers[0][i],
                                               n,
                                               dmm->div_p_lookup_table,
                                               dmm->div_q_lookup_table);
            output_buffers[0][i] =
                utils::compute_sqrt_fingerprint(output_buffers[0][i],
                                                dmm->sqrt_p_lookup_table,
                                                dmm->sqrt_q_lookup_table);
          }
        }
        break;
      }
      default: {
        assert(false && "Unsupported threadblock operator for fingerprinting");
      }
    }
  };

  auto compute_input_operator = [&](threadblock::TBInputOp const *op,
                                    char *smem_buffer,
                                    char *dmem_buffer,
                                    std::vector<int> block_idx,
                                    int forloop_iteration) {
    DTensor const &dtensor = op->dtensor;
    STensor const &stensor = op->output_tensors[0];
    FPType *dtensor_ptr =
        reinterpret_cast<FPType *>(dmem_buffer + dtensor.fp_offset);
    FPType *stensor_ptr =
        reinterpret_cast<FPType *>(smem_buffer + stensor.smem_offset);
    std::vector<size_t> offsets(dtensor.num_dims, 0);

    for (int d = 0; d < 3; d++) {
      int dim_idx = to_vector(op->input_map)[d];
      int dim_div = to_vector(bgraph.grid_dim)[d];
      if (dim_idx >= 0) {
        offsets[dim_idx] = block_idx[d];
      }
    }

    if (op->forloop_dim >= 0) {
      offsets[op->forloop_dim] =
          offsets[op->forloop_dim] * bgraph.forloop_range + forloop_iteration;
    }

    for (size_t i = 0; i < stensor.num_dims; ++i) {
      offsets[i] *= stensor.dim[i];
    }

    for (size_t stensor_idx = 0; stensor_idx < stensor.num_elements();
         ++stensor_idx) {
      std::vector<size_t> stensor_indices =
          linear_idx_to_nd(stensor_idx,
                           op->output_tensors[0].num_dims,
                           op->output_tensors[0].dim);
      std::vector<size_t> dtensor_indices =
          elementwise_add(offsets, stensor_indices);
      size_t dtensor_idx =
          nd_idx_to_linear(dtensor_indices, dtensor.num_dims, dtensor.dim);
      assert(dtensor_idx < dtensor.num_elements());
      stensor_ptr[stensor_idx] = dtensor_ptr[dtensor_idx];
    }
  };

  auto compute_output_operator = [&](threadblock::TBOutputOp const *op,
                                     char *smem_buffer,
                                     char *dmem_buffer,
                                     std::vector<int> block_idx) {
    DTensor const &dtensor = op->dtensor;
    STensor const &stensor = op->input_tensors[0];
    FPType *dtensor_ptr =
        reinterpret_cast<FPType *>(dmem_buffer + dtensor.fp_offset);
    FPType *stensor_ptr =
        reinterpret_cast<FPType *>(smem_buffer + stensor.smem_offset);
    std::vector<size_t> offsets(dtensor.num_dims, 0);

    for (int d = 0; d < 3; d++) {
      int dim_idx = to_vector(op->output_map)[d];
      int dim_div = to_vector(bgraph.grid_dim)[d];
      if (dim_idx >= 0) {
        offsets[dim_idx] = block_idx[d];
      }
    }

    for (size_t i = 0; i < stensor.num_dims; ++i) {
      offsets[i] *= stensor.dim[i];
    }

    for (size_t stensor_idx = 0; stensor_idx < stensor.num_elements();
         ++stensor_idx) {
      std::vector<size_t> stensor_indices = linear_idx_to_nd(
          stensor_idx, op->input_tensors[0].num_dims, op->input_tensors[0].dim);
      std::vector<size_t> dtensor_indices =
          elementwise_add(offsets, stensor_indices);
      size_t dtensor_idx =
          nd_idx_to_linear(dtensor_indices, dtensor.num_dims, dtensor.dim);
      assert(dtensor_idx < dtensor.num_elements());
      dtensor_ptr[dtensor_idx] = stensor_ptr[stensor_idx];
    }
  };

#pragma omp parallel
  {
    for (int device_id = 0; device_id < kgraph->gpu_dim.x; ++device_id) {
      char *dmem_buffer = dmm->fp_base_ptr[device_id];
#pragma omp for collapse(3)
      for (int block_idx_x = 0; block_idx_x < bgraph.grid_dim.x;
           ++block_idx_x) {
        for (int block_idx_y = 0; block_idx_y < bgraph.grid_dim.y;
             ++block_idx_y) {
          for (int block_idx_z = 0; block_idx_z < bgraph.grid_dim.z;
               ++block_idx_z) {
            int thread_block_idx =
                block_idx_x * bgraph.grid_dim.y * bgraph.grid_dim.z +
                block_idx_y * bgraph.grid_dim.z + block_idx_z;
            char *smem_buffer = dmm->stensor_fp_base_ptr +
                                thread_block_idx * config::MAX_SMEM_FP_SIZE;
            for (int forloop_iteration = 0;
                 forloop_iteration < bgraph.forloop_range;
                 ++forloop_iteration) {
              for (TBOperator const *op : bgraph.operators) {
                if (op->op_type == mirage::type::TB_INPUT_OP) {
                  compute_input_operator(
                      static_cast<mirage::threadblock::TBInputOp const *>(op),
                      smem_buffer,
                      dmem_buffer,
                      {block_idx_x, block_idx_y, block_idx_z},
                      forloop_iteration);
                } else if (op->op_type == mirage::type::TB_OUTPUT_OP) {
                  compute_output_operator(
                      static_cast<mirage::threadblock::TBOutputOp const *>(op),
                      smem_buffer,
                      dmem_buffer,
                      {block_idx_x, block_idx_y, block_idx_z});
                } else {
                  compute_operator(op,
                                   smem_buffer,
                                   forloop_iteration == 0,
                                   forloop_iteration ==
                                       bgraph.forloop_range - 1);
                }
              }
            }
          }
        }
      }
    }
  }
}
#endif

} // namespace kernel
} // namespace mirage
