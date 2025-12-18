#include "mirage/search/verification/formal_verifier.h"
#include "mirage/search/op_utils.h"
#include "mirage/search/symbolic_graph/op_args.h"
#include "mirage/search/symbolic_graph/symbolic_graph.h"
#include "mirage/search/symbolic_graph/symbolic_op.h"
#include "mirage/search/symbolic_graph/symbolic_tensor.h"
#include "mirage/search/verification/output_match.h"
#include "mirage/type.h"
#include "mirage/utils/containers.h"

#include <iostream>
#include <unordered_map>
#include <vector>

namespace mirage {
namespace search {

FormalVerifier::FormalVerifier(kernel::Graph const &input_graph) {
  for (kernel::KNOperator *op : input_graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_OUTPUT_OP) {
      shapes_std.push_back(
          to_vector(op->input_tensors[0].num_dims, op->input_tensors[0].dim));
    }
  }
  input_exprs = get_concrete_exprs(input_graph, true);
}

OutputMatch FormalVerifier::verify(kernel::Graph const &graph) {

  std::vector<std::vector<int>> shapes;
  auto is_output_tensor = [&](kernel::DTensor const &dtensor) {
    for (kernel::KNOperator const *op : graph.operators) {
      for (kernel::DTensor const &input_tensor : op->input_tensors) {
        if (input_tensor.guid == dtensor.guid) {
          return false;
        }
      }
    }
    return true;
  };
  for (kernel::KNOperator *op : graph.operators) {
    for (kernel::DTensor dtensor : op->output_tensors) {
      if (is_output_tensor(dtensor)) {
        shapes.push_back(to_vector(dtensor.num_dims, dtensor.dim));
      }
    }
  }
  assert(shapes.size() == shapes_std.size());
  std::vector<std::string> graph_exprs =
      get_concrete_exprs(graph, false);
  assert(input_exprs.size() == graph_exprs.size());

  auto verify_with_match = [&](OutputMatch const &match) {
    for (size_t i = 0; i < match.size(); i++) {
      if (shapes_std[i] != shapes[match[i]]) {
        return false;
      }
      bool is_equiv =
          check_equiv(input_exprs[i].c_str(), graph_exprs[match[i]].c_str(), false);
      if (!is_equiv) {
        return false;
      }
    }
    return true;
  };

  OutputMatch match(input_exprs.size());
  do {
    if (verify_with_match(match)) {
      return match;
    }
  } while (match.next());
  return OutputMatch::invalid_match();
}

OutputMatch FormalVerifier::verify_symbolic_graph(SymbolicKNGraph const &graph) {
  assert(input_exprs.size() == 1);
  std::vector<SymbolicDTensor> output_tensors{graph.tensors.back()};
  std::vector<std::string> graph_exprs = get_concrete_exprs(graph, false);
  assert(output_tensors.size() == graph_exprs.size());

  auto is_shape_match = [&](std::vector<int> const &shape_std, SymbolicDTensor const &symbolic_tensor) {
    std::vector<int> shape_symbolic;
    for (size_t i = 0; i < symbolic_tensor.dims.size(); i++) {
      shape_symbolic.push_back(get_value_with_all_vars_random(symbolic_tensor.dims[i]));
    }
    return shape_std == shape_symbolic;
  };

  auto verify_with_match = [&](OutputMatch const &match) {
    for (size_t i = 0; i < match.size(); i++) {
      if (!is_shape_match(shapes_std[i], output_tensors[match[i]])) {
        return false;
      }
      bool is_equiv = check_equiv(input_exprs[i].c_str(), graph_exprs[match[i]].c_str(), true);
      if (!is_equiv) {
        return false;
      }
    }
    return true;
  };

  OutputMatch match(input_exprs.size());
  do {
    if (verify_with_match(match)) {
      return match;
    }
  } while (match.next());
  return OutputMatch::invalid_match();
}

std::vector<std::string>
    get_concrete_exprs(kernel::Graph const &graph,
                       bool with_output_ops) {
  std::unordered_map<type::GuidType, std::string> tensor_exprs;

  std::string data_dim0 = "data_dim0";
  std::string data_dim1 = "data_dim1";
  std::string data_dim2 = "data_dim2";

  std::string data_dim[3] = {data_dim0, data_dim1, data_dim2};

  int custom_kernel_id = 0;
  int redtox_id = 0;

  auto calc_stensor_exprs = [&](threadblock::Graph const &graph) {
    std::string dx = "dx" + std::to_string(custom_kernel_id);
    std::string dy = "dy" + std::to_string(custom_kernel_id);
    std::string dz = "dz" + std::to_string(custom_kernel_id);
    std::string df = "df" + std::to_string(custom_kernel_id);
    custom_kernel_id++;
    for (threadblock::TBOperator *op : graph.operators) {
      switch (op->op_type) {
        case type::TBOperatorType::TB_INPUT_OP: {
          threadblock::TBInputOp *input_op =
              static_cast<threadblock::TBInputOp *>(op);
          std::string a = tensor_exprs.at(input_op->dtensor.guid);
          if (graph.grid_dim.x > 1) {
            if (input_op->input_map.x >= 0) {
              size_t axis =
                  input_op->dtensor.num_dims - 1 - input_op->input_map.x;
              a = "(partition " + a + " " + data_dim[axis] + " " + dx + " " +
                  std::to_string(graph.grid_dim.x) + ")";
            } else {
              a = "(replicate " + a + " " + dx + " " +
                  std::to_string(graph.grid_dim.x) + ")";
            }
          }
          if (graph.grid_dim.y > 1) {
            if (input_op->input_map.y >= 0) {
              size_t axis =
                  input_op->dtensor.num_dims - 1 - input_op->input_map.y;
              a = "(partition " + a + " " + data_dim[axis] + " " + dy + " " +
                  std::to_string(graph.grid_dim.y) + ")";
            } else {
              a = "(replicate " + a + " " + dy + " " +
                  std::to_string(graph.grid_dim.y) + ")";
            }
          }
          if (graph.grid_dim.z > 1) {
            if (input_op->input_map.z >= 0) {
              size_t axis =
                  input_op->dtensor.num_dims - 1 - input_op->input_map.z;
              a = "(partition " + a + " " + data_dim[axis] + " " + dz + " " +
                  std::to_string(graph.grid_dim.z) + ")";
            } else {
              a = "(replicate " + a + " " + dz + " " +
                  std::to_string(graph.grid_dim.z) + ")";
            }
          }
          if (graph.forloop_range > 1) {
            if (input_op->forloop_dim >= 0) {
              size_t axis =
                  input_op->dtensor.num_dims - 1 - input_op->forloop_dim;
              a = "(partition " + a + " " + data_dim[axis] + " " + df + " " +
                  std::to_string(graph.forloop_range) + ")";
            } else {
              a = "(replicate " + a + " " + df + " " +
                  std::to_string(graph.forloop_range) + ")";
            }
          }
          tensor_exprs.emplace(op->output_tensors[0].guid, a);
          break;
        }
        case type::TBOperatorType::TB_OUTPUT_OP: {
          threadblock::TBOutputOp *output_op =
              static_cast<threadblock::TBOutputOp *>(op);
          std::string a = tensor_exprs.at(output_op->input_tensors[0].guid);
          if (graph.grid_dim.x > 1) {
            if (output_op->output_map.x >= 0) {
              size_t axis = output_op->input_tensors[0].num_dims - 1 -
                            output_op->output_map.x;
              a = "(combine " + a + " " + data_dim[axis] + " " + dx + ")";
            } else {
              a = "(reduce " + a + " " + dx + ")";
            }
          }
          if (graph.grid_dim.y > 1) {
            if (output_op->output_map.y >= 0) {
              size_t axis = output_op->input_tensors[0].num_dims - 1 -
                            output_op->output_map.y;
              a = "(combine " + a + " " + data_dim[axis] + " " + dy + ")";
            } else {
              a = "(reduce " + a + " " + dy + ")";
            }
          }
          if (graph.grid_dim.z > 1) {
            if (output_op->output_map.z >= 0) {
              size_t axis = output_op->input_tensors[0].num_dims - 1 -
                            output_op->output_map.z;
              a = "(combine " + a + " " + data_dim[axis] + " " + dz + ")";
            } else {
              a = "(reduce " + a + " " + dz + ")";
            }
          }
          tensor_exprs.emplace(output_op->dtensor.guid, a);
          break;
        }
        case type::TBOperatorType::TB_ADD_OP: {
          std::string lhs = tensor_exprs.at(op->input_tensors[0].guid);
          std::string rhs = tensor_exprs.at(op->input_tensors[1].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               "(ew_add " + lhs + " " + rhs + ")");
          break;
        }
        case type::TBOperatorType::TB_CONCAT_0_OP:
        case type::TBOperatorType::TB_CONCAT_1_OP:
        case type::TBOperatorType::TB_CONCAT_2_OP: {
          size_t axis = op->input_tensors[0].num_dims - 1 -
                        (op->op_type - type::TBOperatorType::TB_CONCAT_0_OP);
          std::string lhs = tensor_exprs.at(op->input_tensors[0].guid);
          std::string rhs = tensor_exprs.at(op->input_tensors[1].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               "(concat " + lhs + " " + rhs + " " +
                                   data_dim[axis] + ")");
          break;
        }
        case type::TBOperatorType::TB_DIV_OP: {
          std::string lhs = tensor_exprs.at(op->input_tensors[0].guid);
          std::string rhs = tensor_exprs.at(op->input_tensors[1].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               "(bc_div " + lhs + " " + rhs + ")");
          break;
        }
        case type::TBOperatorType::TB_POW_OP: {
          std::string base = tensor_exprs.at(op->input_tensors[0].guid);
          std::string exponent = tensor_exprs.at(op->input_tensors[1].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               "(bc_pow " + base + " " + exponent + ")");
          break;
        }
        case type::TBOperatorType::TB_EXP_OP: {
          std::string a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               "(ew_exp " + a + ")");
          break;
        }
        case type::TBOperatorType::TB_FORLOOP_ACCUM_NO_RED_OP: {
          std::string a = tensor_exprs.at(op->input_tensors[0].guid);
          if (graph.forloop_range > 1) {
            a = "(reduce " + a + " " + df + ")";
          }
          tensor_exprs.emplace(op->output_tensors[0].guid, a);
          break;
        }
        case type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_RMS_OP: {
          std::string a = tensor_exprs.at(op->input_tensors[0].guid);
          // square
          a = "(square " + a + ")";
          // reduce
          if (graph.forloop_range > 1) {
            a = "(reduce " + a + " " + df + ")";
          }
          a = "(sum " + a + " " + data_dim[0] + ")";
          tensor_exprs.emplace(op->output_tensors[0].guid, "(sqrt " + a + ")");
          break;
        }
        case type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_SUM_OP: {
          std::string a = tensor_exprs.at(op->input_tensors[0].guid);
          if (graph.forloop_range > 1) {
            a = "(reduce " + a + " " + df + ")";
          }
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               "(sum " + a + " " + data_dim[0] + ")");
          break;
        }
        case type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_MEAN_OP: {
          std::string a = tensor_exprs.at(op->input_tensors[0].guid);
          if (graph.forloop_range > 1) {
            a = "(reduce " + a + " " + df + ")";
          }
          tensor_exprs.emplace(op->output_tensors[0].guid, "(mean " + a + ")");
          break;
        }
        case type::TBOperatorType::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP: {
          std::string a = tensor_exprs.at(op->input_tensors[0].guid);
          if (graph.forloop_range > 1) {
            a = "(reduce " + a + " " + df + ")";
          }
          std::string reddim = "reddim" + std::to_string(redtox_id++);
          int reduce_degree =
              op->input_tensors[0].dim[op->input_tensors[0].num_dims - 1] /
              op->output_tensors[0].dim[op->output_tensors[0].num_dims - 1];
          a = "(partition " + a + " " + data_dim[0] + " " + reddim + " " +
              std::to_string(reduce_degree) + ")";
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               "(reduce " + a + " " + reddim + ")");
          break;
        }
        case type::TBOperatorType::TB_MATMUL_OP: {
          std::string lhs = tensor_exprs.at(op->input_tensors[0].guid);
          std::string rhs = tensor_exprs.at(op->input_tensors[1].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               "(matmul " + lhs + " " + rhs + ")");
          break;
        }
        case type::TBOperatorType::TB_MUL_OP: {
          std::string lhs = tensor_exprs.at(op->input_tensors[0].guid);
          std::string rhs = tensor_exprs.at(op->input_tensors[1].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               "(ew_mul " + lhs + " " + rhs + ")");
          break;
        }
        case type::TBOperatorType::TB_MUL_SCALAR_OP: {
          assert(false && "TODO");
          break;
        }
        case type::TBOperatorType::TB_REDUCTION_0_OP:
        case type::TBOperatorType::TB_REDUCTION_1_OP:
        case type::TBOperatorType::TB_REDUCTION_2_OP: {
          size_t axis = op->input_tensors[0].num_dims - 1 -
                        (op->op_type - type::TBOperatorType::TB_REDUCTION_0_OP);
          std::string a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               "(sum " + a + " " + data_dim[axis] + ")");
          break;
        }
        case type::TBOperatorType::TB_RMS_NORM_OP: {
          std::string a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               "(rms_norm " + a + " " + data_dim[0] + ")");
          break;
        }
        case type::TBOperatorType::TB_SQUARE_OP: {
          std::string a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               "(square " + a + ")");
          break;
        }
        case type::TBOperatorType::TB_SQRT_OP: {
          std::string a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, "(sqrt " + a + ")");
          break;
        }
        case type::TBOperatorType::TB_SILU_OP: {
          std::string a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, "(silu " + a + ")");
          break;
        }
        case type::TBOperatorType::TB_GELU_OP: {
          std::string a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, "(gelu " + a + ")");
          break;
        }
        case type::TBOperatorType::TB_RELU_OP: {
          std::string a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, "(relu " + a + ")");
          break;
        }
        case type::TBOperatorType::TB_CLAMP_OP: {
          std::string a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, "(clamp " + a + ")");
          break;
        }
        default:
          assert(false && "Unsupported operator type");
      }
    }
  };

  auto calc_dtensor_exprs = [&](kernel::Graph const &graph) {
    int input_id = 0;
    for (kernel::KNOperator *op : graph.operators) {
      switch (op->op_type) {
        case type::KNOperatorType::KN_INPUT_OP: {
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               "input" + std::to_string(input_id++));
          break;
        }
        case type::KNOperatorType::KN_OUTPUT_OP: {
          break;
        }
        case type::KNOperatorType::KN_ADD_OP: {
          std::string lhs = tensor_exprs.at(op->input_tensors[0].guid);
          std::string rhs = tensor_exprs.at(op->input_tensors[1].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               "(ew_add " + lhs + " " + rhs + ")");
          break;
        }
        case type::KNOperatorType::KN_DIV_OP: {
          std::string lhs = tensor_exprs.at(op->input_tensors[0].guid);
          std::string rhs = tensor_exprs.at(op->input_tensors[1].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               "(bc_div " + lhs + " " + rhs + ")");
          break;
        }
        case type::KNOperatorType::KN_POW_OP: {
          std::string base = tensor_exprs.at(op->input_tensors[0].guid);
          std::string exponent = tensor_exprs.at(op->input_tensors[1].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               "(bc_pow " + base + " " + exponent + ")");
          break;
        }
        case type::KNOperatorType::KN_EXP_OP: {
          std::string a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               "(ew_exp " + a + ")");
          break;
        }
        case type::KNOperatorType::KN_SQUARE_OP: {
          std::string a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               "(square " + a + ")");
          break;
        }
        case type::KNOperatorType::KN_SQRT_OP: {
          std::string a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, "(sqrt " + a + ")");
          break;
        }
        case type::KNOperatorType::KN_MATMUL_OP: {
          std::string lhs = tensor_exprs.at(op->input_tensors[0].guid);
          std::string rhs = tensor_exprs.at(op->input_tensors[1].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               "(matmul " + lhs + " " + rhs + ")");
          break;
        }
        case type::KNOperatorType::KN_MUL_OP: {
          std::string lhs = tensor_exprs.at(op->input_tensors[0].guid);
          std::string rhs = tensor_exprs.at(op->input_tensors[1].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               "(ew_mul " + lhs + " " + rhs + ")");
          break;
        }
        case type::KNOperatorType::KN_REDUCTION_0_OP:
        case type::KNOperatorType::KN_REDUCTION_1_OP:
        case type::KNOperatorType::KN_REDUCTION_2_OP: {
          size_t axis = op->input_tensors[0].num_dims - 1 -
                        (op->op_type - type::KNOperatorType::KN_REDUCTION_0_OP);
          std::string a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               "(sum " + a + " " + data_dim[axis] + ")");
          break;
        }
        case type::KNOperatorType::KN_RMS_NORM_OP: {
          std::string a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid,
                               "(rms_norm " + a + " " + data_dim[0] + ")");
          break;
        }
        case type::KNOperatorType::KN_SILU_OP: {
          std::string a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, "(silu " + a + ")");
          break;
        }
        case type::KNOperatorType::KN_GELU_OP: {
          std::string a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, "(gelu " + a + ")");
          break;
        }
        case type::KNOperatorType::KN_RELU_OP: {
          std::string a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, "(relu " + a + ")");
          break;
        }
        case type::KNOperatorType::KN_CLAMP_OP: {
          std::string a = tensor_exprs.at(op->input_tensors[0].guid);
          tensor_exprs.emplace(op->output_tensors[0].guid, "(clamp " + a + ")");
          break;
        }
        case type::KNOperatorType::KN_CUSTOMIZED_OP: {
          calc_stensor_exprs(static_cast<kernel::KNCustomizedOp *>(op)->bgraph);
          break;
        }
        default:
          assert(false && "Unsupported operator type");
      }
    }
  };

  calc_dtensor_exprs(graph);

  std::vector<std::string> output_exprs;

  if (with_output_ops) {
    for (kernel::KNOperator *op : graph.operators) {
      if (op->op_type == type::KNOperatorType::KN_OUTPUT_OP) {
        output_exprs.push_back(tensor_exprs.at(op->input_tensors[0].guid));
      }
    }
  } else {
    for (kernel::KNOperator *op : graph.operators) {
      for (kernel::DTensor const &tensor : op->output_tensors) {
        if (get_num_consumers(graph, tensor) == 0) {
          output_exprs.push_back(tensor_exprs.at(tensor.guid));
        }
      }
    }
  }
  return output_exprs;
}

std::vector<std::string> get_concrete_exprs(SymbolicKNGraph const &graph, bool with_output_ops) {
  std::string data_dim0 = "data_dim0";
  std::string data_dim1 = "data_dim1";
  std::string data_dim2 = "data_dim2";

  std::string data_dim[3] = {data_dim0, data_dim1, data_dim2};

  auto calc_stensor_exprs = [&](SymbolicTBGraph const &graph, std::vector<std::string> const &inputs) {
    std::vector<std::string> tensor_exprs, output_dtensor_exprs;

    bool is_forloop_greater_than_one = [&] {
      for (auto const &op : graph.operators) {
        if (op.op_type == type::TB_INPUT_OP) {
          if (std::static_pointer_cast<TBInputOpArgs const>(op.args)->forloop_dim >= 0) {
            return true;
          }
        }
      }
      return false;
    }();

    std::vector<std::string> parallel_dim = vector_map(graph.grid_dim, [&](SymbolicTensorDim const &dim) {
      assert(dim->is_var());
      return dim->to_string();
    });
    std::vector<int> dummy_dim_size = vector_map(graph.grid_dim, [&](SymbolicTensorDim const &dim) {
      assert(dim->is_var());
      return (int)std::static_pointer_cast<TensorDimVar const>(dim)->index;
    });
    std::string forloop_range = graph.forloop_range->to_string();
    int dummy_fsize = std::static_pointer_cast<TensorDimVar const>(graph.forloop_range)->index;
    for (size_t i = 0; i < graph.operators.size(); ++i) {
      switch (graph.operators[i].op_type) {
        case type::TBOperatorType::TB_INPUT_OP: {
          auto args = std::static_pointer_cast<TBInputOpArgs const>(graph.operators[i].args);
          std::string a = inputs[i];
          for (size_t j = 0; j < args->input_map.size(); ++j) {
            if (args->input_map[j] >= 0) {
              size_t axis = args->dtensor.dims.size() - 1 - args->input_map[j];
              a = "(partition " + a + " " + data_dim[axis] + " " + parallel_dim[j] + " " + std::to_string(dummy_dim_size[j]) + ")";
            } else {
              a = "(replicate " + a + " " + parallel_dim[j] + " " + std::to_string(dummy_dim_size[j]) + ")";
            }
          }
          if (is_forloop_greater_than_one) {
            if (args->forloop_dim >= 0) {
              size_t axis = args->dtensor.dims.size() - 1 - args->forloop_dim;
              a = "(partition " + a + " " + data_dim[axis] + " " + forloop_range + " " + std::to_string(dummy_fsize) + ")";
            } else {
              a = "(replicate " + a + " " + forloop_range + " " + std::to_string(dummy_fsize) + ")";
            }
          }
          tensor_exprs.push_back(a);
          break;
        }
        case type::TBOperatorType::TB_OUTPUT_OP: {
          auto args = std::static_pointer_cast<TBOutputOpArgs const>(graph.operators[i].args);
          std::string a = tensor_exprs[graph.input_indices[i][0]];
          // int3 omap = vec_to_int3(args->output_map);
          for (size_t j = 0; j < args->output_map.size(); ++j) {
            if (args->output_map[j] >= 0) {
              size_t axis = args->dtensor.dims.size() - 1 - args->output_map[j];
              a = "(combine " + a + " " + data_dim[axis] + " " + parallel_dim[j] + ")";
            } else {
              a = "(reduce " + a + " " + parallel_dim[j] + ")";
            }
          }
          output_dtensor_exprs.push_back(a);
          break;
        }
        case type::TBOperatorType::TB_ADD_OP: {
          std::string lhs = tensor_exprs[graph.input_indices[i][0]];
          std::string rhs = tensor_exprs[graph.input_indices[i][1]];
          std::string a = "(ew_add " + lhs + " " + rhs + ")";
          tensor_exprs.push_back(a);
          break;
        }
        case type::TBOperatorType::TB_DIV_OP: {
          std::string lhs = tensor_exprs[graph.input_indices[i][0]];
          std::string rhs = tensor_exprs[graph.input_indices[i][1]];
          std::string a = "(bc_div " + lhs + " " + rhs + ")";
          tensor_exprs.push_back(a);
          break;
        }
        case type::TBOperatorType::TB_POW_OP: {
          std::string base = tensor_exprs[graph.input_indices[i][0]];
          std::string exponent = tensor_exprs[graph.input_indices[i][1]];
          std::string a = "(bc_pow " + base + " " + exponent + ")";
          tensor_exprs.push_back(a);
          break;
        }
        case type::TBOperatorType::TB_EXP_OP: {
          std::string a = tensor_exprs[graph.input_indices[i][0]];
          a = "(ew_exp " + a + ")";
          tensor_exprs.push_back(a);
          break;
        }
        case type::TBOperatorType::TB_FORLOOP_ACCUM_NO_RED_OP: {
          std::string a = tensor_exprs[graph.input_indices[i][0]];
          if (is_forloop_greater_than_one) {
            a = "(reduce " + a + " " + forloop_range + ")";
          }
          tensor_exprs.push_back(a);
          break;
        }
        case type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_RMS_OP: {
          std::string a = tensor_exprs[graph.input_indices[i][0]];
          a = "(square " + a + ")";
          if (is_forloop_greater_than_one) {
            a = "(reduce " + a + " " + forloop_range + ")";
          }
          a = "(sum " + a + " " + data_dim[0] + ")";
          a = "(sqrt " + a + ")";
          tensor_exprs.push_back(a);
          break;
        }
        case type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_SUM_OP: {
          std::string a = tensor_exprs[graph.input_indices[i][0]];
          if (is_forloop_greater_than_one) {
            a = "(reduce " + a + " " + forloop_range + ")";
          }
          a = "(sum " + a + " " + data_dim[0] + ")";
          tensor_exprs.push_back(a);
          break;
        }
        case type::TBOperatorType::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP: {
          std::string a = tensor_exprs[graph.input_indices[i][0]];
          if (is_forloop_greater_than_one) {
            a = "(reduce " + a + " " + forloop_range + ")";
          }
          // std::string reddim = "reddim" + std::to_string(redtox_id++);
          assert(graph.reduction_degree);
          assert(graph.reduction_degree->is_var());
          std::string reddim = graph.reduction_degree->to_string();
          int reduce_degree = (int)std::static_pointer_cast<TensorDimVar const>(graph.reduction_degree)->index;
          a = "(partition " + a + " " + data_dim[0] + " " + reddim + " " +
              std::to_string(reduce_degree) + ")";
          a = "(reduce " + a + " " + reddim + ")";
          tensor_exprs.push_back(a);
          break;
        }
        case type::TBOperatorType::TB_MATMUL_OP: {
          std::string lhs = tensor_exprs[graph.input_indices[i][0]];
          std::string rhs = tensor_exprs[graph.input_indices[i][1]];
          std::string a = "(matmul " + lhs + " " + rhs + ")";
          tensor_exprs.push_back(a);
          break;
        }
        case type::TBOperatorType::TB_MUL_OP: {
          std::string lhs = tensor_exprs[graph.input_indices[i][0]];
          std::string rhs = tensor_exprs[graph.input_indices[i][1]];
          std::string a = "(ew_mul " + lhs + " " + rhs + ")";
          tensor_exprs.push_back(a);
          break;
        }
        case type::TBOperatorType::TB_RMS_NORM_OP: {
          std::string a = tensor_exprs[graph.input_indices[i][0]];
          a = "(rms_norm " + a + " " + data_dim[0] + ")";
          tensor_exprs.push_back(a);
          break;
        }
        case type::TBOperatorType::TB_SQUARE_OP: {
          std::string a = tensor_exprs[graph.input_indices[i][0]];
          a = "(square " + a + ")";
          tensor_exprs.push_back(a);
          break;
        }
        case type::TBOperatorType::TB_SQRT_OP: {
          std::string a = tensor_exprs[graph.input_indices[i][0]];
          a = "(sqrt " + a + ")";
          tensor_exprs.push_back(a);
          break;
        }
        case type::TBOperatorType::TB_SILU_OP: {
          std::string a = tensor_exprs[graph.input_indices[i][0]];
          a = "(silu " + a + ")";
          tensor_exprs.push_back(a);
          break;
        }
        case type::TBOperatorType::TB_GELU_OP: {
          std::string a = tensor_exprs[graph.input_indices[i][0]];
          a = "(gelu " + a + ")";
          tensor_exprs.push_back(a);
          break;
        }
        case type::TBOperatorType::TB_RELU_OP: {
          std::string a = tensor_exprs[graph.input_indices[i][0]];
          a = "(relu " + a + ")";
          tensor_exprs.push_back(a);
          break;
        }
        case type::TBOperatorType::TB_CLAMP_OP: {
          std::string a = tensor_exprs[graph.input_indices[i][0]];
          a = "(clamp " + a + ")";
          tensor_exprs.push_back(a);
          break;
        }
        default: {
          std::cerr << "Unsupported operator type: " << json(graph.operators[i].op_type) << std::endl;
          assert(false && "Unsupported operator type");
        }
      }
    }
    return output_dtensor_exprs;
  };

  auto calc_dtensor_exprs = [&](SymbolicKNGraph const &graph) {
    std::vector<std::string> tensor_exprs;
    int input_id = 0;
    for (size_t i = 0; i < graph.operators.size(); ++i) {
      switch (graph.operators[i].op_type) {
        case type::KNOperatorType::KN_INPUT_OP: {
          std::string a = "input" + std::to_string(input_id++);
          tensor_exprs.push_back(a);
          break;
        }
        case type::KNOperatorType::KN_OUTPUT_OP: {
          break;
        }
        case type::KNOperatorType::KN_CUSTOMIZED_OP: {
          std::vector<std::string> input_exprs = vector_map(graph.input_indices[i], [&](int i) { return tensor_exprs[i]; });
          std::vector<std::string> output_exprs = calc_stensor_exprs(std::static_pointer_cast<KNCustomizedOpArgs const>(graph.operators[i].args)->tb_graph_template, input_exprs);
          tensor_exprs.insert(tensor_exprs.end(), output_exprs.begin(), output_exprs.end());
          break;
        }
        default: {
          assert(false && "Unsupported operator type");
        }
      }
    }
    return std::vector<std::string>{tensor_exprs.back()};
  };

  std::vector<std::string> tensor_exprs = calc_dtensor_exprs(graph);
  return tensor_exprs;
}

} // namespace search
} // namespace mirage
