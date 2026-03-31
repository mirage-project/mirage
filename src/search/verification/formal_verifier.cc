#include "mirage/search/verification/formal_verifier.h"
#include "mirage/search/op_utils.h"
#include "mirage/search/symbolic_graph/op_args.h"
#include "mirage/search/symbolic_graph/symbolic_graph.h"
#include "mirage/search/symbolic_graph/symbolic_op.h"
#include "mirage/search/symbolic_graph/symbolic_tensor.h"
#include "mirage/search/verification/output_match.h"
#include "mirage/type.h"
#include "mirage/utils/containers.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <set>
#include <tuple>
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
  bool with_output_ops = false;
  for (kernel::KNOperator *op : graph.operators) {
    if (op->op_type == type::KNOperatorType::KN_OUTPUT_OP) {
      shapes.push_back(to_vector(op->input_tensors[0].num_dims, op->input_tensors[0].dim));
      with_output_ops = true;
      continue;
    }
    for (kernel::DTensor dtensor : op->output_tensors) {
      if (is_output_tensor(dtensor)) {
        assert(!with_output_ops);
        shapes.push_back(to_vector(dtensor.num_dims, dtensor.dim));
      }
    }
  }
  assert(shapes.size() == shapes_std.size());
  std::vector<std::string> graph_exprs =
      get_concrete_exprs(graph, with_output_ops);
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
      shape_symbolic.push_back(get_value_with_bool_vars_zero_others_random(symbolic_tensor.dims[i]));
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
          if (std::static_pointer_cast<TBInputOpArgs const>(op.args)->input_map.to_legacy_forloop_dim() >= 0) {
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
      return (int)std::static_pointer_cast<TensorDimVar const>(dim)->index + 2;
    });
    std::string forloop_range = graph.forloop_range->to_string();
    int dummy_fsize = std::static_pointer_cast<TensorDimVar const>(graph.forloop_range)->index + 2;
    for (size_t i = 0; i < graph.operators.size(); ++i) {
      switch (graph.operators[i].op_type) {
        case type::TBOperatorType::TB_INPUT_OP: {
          auto args = std::static_pointer_cast<TBInputOpArgs const>(graph.operators[i].args);
          std::string a = inputs[i];
          std::vector<int> legacy_imap = args->input_map.to_legacy_map();
          int legacy_forloop = args->input_map.to_legacy_forloop_dim();
          for (size_t j = 0; j < legacy_imap.size(); ++j) {
            if (legacy_imap[j] >= 0) {
              size_t axis = args->dtensor.dims.size() - 1 - legacy_imap[j];
              a = "(partition " + a + " " + data_dim[axis] + " " + parallel_dim[j] + " " + std::to_string(dummy_dim_size[j]) + ")";
            } else {
              a = "(replicate " + a + " " + parallel_dim[j] + " " + std::to_string(dummy_dim_size[j]) + ")";
            }
          }
          if (is_forloop_greater_than_one) {
            if (legacy_forloop >= 0) {
              size_t axis = args->dtensor.dims.size() - 1 - legacy_forloop;
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
          std::vector<int> legacy_omap = args->output_map.to_legacy_map();
          for (size_t j = 0; j < legacy_omap.size(); ++j) {
            if (legacy_omap[j] >= 0) {
              size_t axis = args->dtensor.dims.size() - 1 - legacy_omap[j];
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

// Rebuild a SymbolicTBGraph replacing its symbolic maps with concrete ones.
// input_maps[i] is the legacy map for the i-th TB_INPUT_OP,
// forloop_dims[i] is the forloop_dim for the i-th TB_INPUT_OP,
// output_maps[i] is the legacy map for the i-th TB_OUTPUT_OP.
static std::optional<SymbolicTBGraph> rebuild_tb_with_concrete_maps(
    SymbolicTBGraph const &tb,
    std::vector<std::vector<int>> const &input_maps,
    std::vector<int> const &forloop_dims,
    std::vector<std::vector<int>> const &output_maps,
    int num_grid_dims_override = -1) {
  int actual_num_grid_dims = (num_grid_dims_override > 0)
      ? num_grid_dims_override
      : static_cast<int>(tb.grid_dim.size());
  SymbolicTBGraph result(tb.dim_variable_index_base, actual_num_grid_dims);
  // Use first actual_num_grid_dims grid dims from tb
  result.grid_dim.assign(tb.grid_dim.begin(),
                         tb.grid_dim.begin() + std::min((int)tb.grid_dim.size(), actual_num_grid_dims));
  // If collapsed to fewer dims, pad grid_dim if needed (shouldn't happen)
  while ((int)result.grid_dim.size() < actual_num_grid_dims) {
    result.grid_dim.push_back(dim_expr_make_const(1));
  }
  result.block_dim = tb.block_dim;
  result.forloop_range = tb.forloop_range;
  result.reduction_degree = tb.reduction_degree;
  result.next_dim_variable_index = tb.next_dim_variable_index;

  size_t input_idx = 0, output_idx = 0;
  for (size_t i = 0; i < tb.operators.size(); ++i) {
    auto const &op = tb.operators[i];
    if (op.op_type == type::TBOperatorType::TB_INPUT_OP) {
      auto args = static_cast<TBInputOpArgs const *>(op.args.get());
      if (!result.add_input(args->dtensor, input_maps[input_idx], forloop_dims[input_idx])) {
        return std::nullopt;
      }
      ++input_idx;
    } else if (op.op_type == type::TBOperatorType::TB_OUTPUT_OP) {
      auto args = static_cast<TBOutputOpArgs const *>(op.args.get());
      int tb_input_index = tb.input_indices[i][0];
      if (!result.add_output(tb_input_index, output_maps[output_idx], args->epilogue)) {
        return std::nullopt;
      }
      ++output_idx;
    } else {
      if (!result.add_operator(op.op_type, tb.input_indices[i])) {
        return std::nullopt;
      }
    }
  }
  return result;
}

// Rebuild a SymbolicKNGraph replacing a specific KN_CUSTOMIZED_OP's TB graph.
static SymbolicKNGraph rebuild_kn_with_tb(
    SymbolicKNGraph const &kn,
    size_t customized_op_idx,
    SymbolicTBGraph const &new_tb) {
  SymbolicKNGraph result;
  result.next_dim_variable_index = kn.next_dim_variable_index;

  for (size_t i = 0; i < kn.operators.size(); ++i) {
    auto const &op = kn.operators[i];
    auto const &ref_inputs = kn.input_indices[i];

    if (op.op_type == type::KNOperatorType::KN_CUSTOMIZED_OP && i == customized_op_idx) {
      result.add_customized_operator(new_tb, ref_inputs);
    } else if (op.op_type == type::KNOperatorType::KN_INPUT_OP) {
      auto args = std::static_pointer_cast<KNInputOpArgs const>(op.args);
      result.add_input(args->input_dims, args->input_strides, args->data_type, args->layout, args->input_map);
    } else if (op.op_type == type::KNOperatorType::KN_OUTPUT_OP) {
      auto args = std::static_pointer_cast<KNOutputOpArgs const>(op.args);
      result.add_output(ref_inputs[0], args->output_strides, args->output_map);
    } else {
      result.add_operator(op.op_type, ref_inputs);
    }
  }
  return result;
}

// Generate maps where each grid dim maps to -1 (replicate) or a unique data dim.
// No two grid dims map to the same data dim within one map.
static void enumerate_maps(int num_dims, int num_data_dims,
                           std::vector<std::vector<int>> &out) {
  std::vector<int> map(num_dims);
  std::vector<bool> used(num_data_dims, false);
  std::function<void(int)> gen = [&](int d) {
    if (d == num_dims) { out.push_back(map); return; }
    map[d] = -1; gen(d + 1);  // replicate
    for (int v = 0; v < num_data_dims; ++v) {
      if (!used[v]) {
        used[v] = true; map[d] = v;
        gen(d + 1);
        used[v] = false;
      }
    }
  };
  gen(0);
}

// A single map assignment for all inputs and outputs.
struct MapCombo {
  std::vector<std::vector<int>> input_maps;  // [num_inputs][num_grid_dims]
  std::vector<int> forloop_dims;             // [num_inputs]
  std::vector<std::vector<int>> output_maps; // [num_outputs][num_grid_dims]
};

// Check if this combo is the lexicographically smallest under grid-dim permutations.
static bool is_canonical_combo(MapCombo const &combo, int num_grid_dims) {
  if (num_grid_dims <= 1) return true;
  std::vector<size_t> perm(num_grid_dims);
  std::iota(perm.begin(), perm.end(), 0);
  while (std::next_permutation(perm.begin(), perm.end())) {
    // Compare input maps
    for (size_t t = 0; t < combo.input_maps.size(); ++t) {
      for (int p = 0; p < num_grid_dims; ++p) {
        int orig = combo.input_maps[t][p];
        int permuted = combo.input_maps[t][perm[p]];
        if (permuted < orig) return false;
        if (permuted > orig) goto next_perm;
      }
    }
    // Compare output maps
    for (size_t t = 0; t < combo.output_maps.size(); ++t) {
      for (int p = 0; p < num_grid_dims; ++p) {
        int orig = combo.output_maps[t][p];
        int permuted = combo.output_maps[t][perm[p]];
        if (permuted < orig) return false;
        if (permuted > orig) goto next_perm;
      }
    }
    next_perm:;
  }
  return true;
}

// Precompute all canonical map combos for given input/output data dims.
static std::vector<MapCombo> precompute_canonical_combos(
    int num_grid_dims,
    std::vector<int> const &input_data_dims,
    std::vector<int> const &output_data_dims) {
  // Enumerate per-slot candidates
  std::vector<std::vector<std::vector<int>>> per_input_maps;
  std::vector<std::vector<int>> per_input_forloops;
  for (int nd : input_data_dims) {
    std::vector<std::vector<int>> grid_maps;
    enumerate_maps(num_grid_dims, nd, grid_maps);
    per_input_maps.push_back(grid_maps);
    std::vector<int> fl_cands;
    for (int d = -1; d < nd; ++d) fl_cands.push_back(d);
    per_input_forloops.push_back(fl_cands);
  }
  std::vector<std::vector<std::vector<int>>> per_output_maps;
  for (int nd : output_data_dims) {
    std::vector<std::vector<int>> grid_maps;
    enumerate_maps(num_grid_dims, nd, grid_maps);
    per_output_maps.push_back(grid_maps);
  }

  // Build all combos via recursive enumeration, filter by canonical
  std::vector<MapCombo> result;
  MapCombo current;
  current.input_maps.resize(input_data_dims.size());
  current.forloop_dims.resize(input_data_dims.size());
  current.output_maps.resize(output_data_dims.size());

  std::function<void(size_t)> enum_inputs;
  std::function<void(size_t)> enum_outputs;

  enum_outputs = [&](size_t oi) {
    if (oi == output_data_dims.size()) {
      if (is_canonical_combo(current, num_grid_dims)) {
        result.push_back(current);
      }
      return;
    }
    for (auto const &omap : per_output_maps[oi]) {
      current.output_maps[oi] = omap;
      enum_outputs(oi + 1);
    }
  };

  // Partial canonicality check on input maps only (before enumerating outputs).
  // If any grid-dim permutation produces a lexicographically smaller input map
  // sequence, the full combo can't be canonical regardless of output maps.
  auto is_canonical_inputs = [&]() {
    if (num_grid_dims <= 1) return true;
    std::vector<size_t> perm(num_grid_dims);
    std::iota(perm.begin(), perm.end(), 0);
    while (std::next_permutation(perm.begin(), perm.end())) {
      for (size_t t = 0; t < current.input_maps.size(); ++t) {
        for (int p = 0; p < num_grid_dims; ++p) {
          int orig = current.input_maps[t][p];
          int permuted = current.input_maps[t][perm[p]];
          if (permuted < orig) return false;
          if (permuted > orig) goto next_perm_input;
        }
      }
      next_perm_input:;
    }
    return true;
  };

  enum_inputs = [&](size_t ii) {
    if (ii == input_data_dims.size()) {
      if (!is_canonical_inputs()) return;
      enum_outputs(0);
      return;
    }
    for (auto const &imap : per_input_maps[ii]) {
      current.input_maps[ii] = imap;
      for (int fl : per_input_forloops[ii]) {
        current.forloop_dims[ii] = fl;
        enum_inputs(ii + 1);
      }
    }
  };

  enum_inputs(0);
  return result;
}

std::vector<std::pair<SymbolicKNGraph, OutputMatch>>
FormalVerifier::verify_symbolic_graph_with_unknown_maps(
    SymbolicKNGraph const &graph) {
  std::vector<std::pair<SymbolicKNGraph, OutputMatch>> results;

  // Shape pre-check: set all map vars to 0 (replicate), check dims
  {
    std::vector<SymbolicDTensor> output_tensors{graph.tensors.back()};
    for (size_t i = 0; i < shapes_std.size(); ++i) {
      std::vector<int> shape_symbolic;
      for (auto const &dim : output_tensors[i % output_tensors.size()].dims) {
        shape_symbolic.push_back(get_value_with_bool_vars_zero_others_random(dim));
      }
      // With all maps=0 (replicate), the output shape should match
      // the "all vars random" shape of the standard output
      // This is a quick sanity check — skip if dims don't even match count
      if (shape_symbolic.size() != shapes_std[i].size()) {
        return results;
      }
    }
  }

  // Find the KN_CUSTOMIZED_OP and its TB graph
  for (size_t op_idx = 0; op_idx < graph.operators.size(); ++op_idx) {
    if (graph.operators[op_idx].op_type != type::KNOperatorType::KN_CUSTOMIZED_OP) {
      continue;
    }
    auto args = std::static_pointer_cast<KNCustomizedOpArgs const>(graph.operators[op_idx].args);
    SymbolicTBGraph const &tb = args->tb_graph_template;

    // Count input/output ops and collect their data dims
    struct InputInfo { int num_data_dims; SymbolicDTensor dtensor; };
    struct OutputInfo { int num_data_dims; };
    std::vector<InputInfo> input_infos;
    std::vector<OutputInfo> output_infos;
    int num_grid_dims = tb.grid_dim.size();

    for (size_t i = 0; i < tb.operators.size(); ++i) {
      if (tb.operators[i].op_type == type::TBOperatorType::TB_INPUT_OP) {
        auto a = static_cast<TBInputOpArgs const *>(tb.operators[i].args.get());
        input_infos.push_back({(int)a->dtensor.dims.size(), a->dtensor});
      } else if (tb.operators[i].op_type == type::TBOperatorType::TB_OUTPUT_OP) {
        auto a = static_cast<TBOutputOpArgs const *>(tb.operators[i].args.get());
        output_infos.push_back({(int)a->dtensor.dims.size()});
      }
    }

    // Extract map equality constraints from dim_equalities.
    // For each (A, B) in dim_equalities, find which input stensor dim
    // matches A and B, then record that their maps must agree.
    // Constraint: (input_i, data_dim_d) must have same map as (input_j, data_dim_e)
    struct MapConstraint {
      int input_i, dim_d;   // input_maps[i][p]==d iff input_maps[j][p]==e
      int input_j, dim_e;
    };
    std::vector<MapConstraint> map_constraints;
    {
      // Collect stensor indices for each input op
      std::vector<int> input_stensor_indices;
      for (size_t i = 0; i < tb.operators.size(); ++i) {
        if (tb.operators[i].op_type == type::TBOperatorType::TB_INPUT_OP) {
          input_stensor_indices.push_back(tb.output_indices[i][0]);
        }
      }

      auto find_input_dim = [&](SymbolicTensorDim const &target) -> std::pair<int, int> {
        for (size_t i = 0; i < input_stensor_indices.size(); ++i) {
          auto const &dims = tb.tensors[input_stensor_indices[i]].dims;
          for (size_t d = 0; d < dims.size(); ++d) {
            if (dims[d]->same_expr_as(target)) {
              return {(int)i, (int)d};
            }
          }
        }
        return {-1, -1};
      };

      for (auto const &[A, B] : tb.dim_equalities) {
        auto [i, d] = find_input_dim(A);
        auto [j, e] = find_input_dim(B);
        if (i >= 0 && j >= 0) {
          map_constraints.push_back({i, d, j, e});
        }
      }
    }

    // Check whether a combo satisfies all map constraints.
    auto satisfies_constraints = [&](MapCombo const &combo) {
      for (auto const &c : map_constraints) {
        // For each grid dim p: (maps[i][p]==d) must equal (maps[j][p]==e)
        for (int p = 0; p < num_grid_dims; ++p) {
          bool i_maps_d = (combo.input_maps[c.input_i][p] == c.dim_d);
          bool j_maps_e = (combo.input_maps[c.input_j][p] == c.dim_e);
          if (i_maps_d != j_maps_e) return false;
        }
        // Forloop: (forloop[i]==d) must equal (forloop[j]==e)
        bool i_fl_d = (combo.forloop_dims[c.input_i] == c.dim_d);
        bool j_fl_e = (combo.forloop_dims[c.input_j] == c.dim_e);
        if (i_fl_d != j_fl_e) return false;
      }
      return true;
    };

    // Precompute canonical map combos with symmetry breaking (cached)
    std::vector<int> input_data_dims;
    for (auto const &info : input_infos) input_data_dims.push_back(info.num_data_dims);
    std::vector<int> output_data_dims;
    for (auto const &info : output_infos) output_data_dims.push_back(info.num_data_dims);

    using CacheKey = std::tuple<int, std::vector<int>, std::vector<int>>;
    static std::mutex combo_cache_mutex;
    static std::map<CacheKey, std::vector<MapCombo>> combo_cache;

    CacheKey key{num_grid_dims, input_data_dims, output_data_dims};
    std::vector<MapCombo> const *combos_ptr;
    {
      std::lock_guard<std::mutex> lock(combo_cache_mutex);
      auto it = combo_cache.find(key);
      if (it == combo_cache.end()) {
        it = combo_cache.emplace(key, precompute_canonical_combos(
            num_grid_dims, input_data_dims, output_data_dims)).first;
      }
      combos_ptr = &it->second;
    }
    auto const &combos = *combos_ptr;

    // Track collapsed map signatures to avoid duplicate verification
    // across different symbolic graph candidates within the same search.
    // Key includes op types + KN input shapes to avoid cross-workload collisions.
    std::vector<int> op_type_seq;
    for (auto const &op : tb.operators) {
      op_type_seq.push_back(static_cast<int>(op.op_type));
    }
    std::vector<std::vector<int>> kn_input_shapes;
    for (auto const &t : graph.tensors) {
      std::vector<int> shape;
      for (auto const &d : t.dims) {
        shape.push_back(get_value_with_bool_vars_zero_others_random(d));
      }
      kn_input_shapes.push_back(shape);
    }
    static std::mutex seen_collapsed_mutex;
    static std::set<std::tuple<std::vector<std::vector<int>>,
                               std::vector<int>,
                               std::vector<std::vector<int>>,
                               std::vector<int>,
                               std::vector<std::vector<int>>>> seen_collapsed;

    for (auto const &combo : combos) {
      if (!satisfies_constraints(combo)) {
        continue;
      }

      // Collapse degenerate grid dims: if grid dim p maps to -1 for all
      // inputs and all outputs, remove it before verification.
      std::vector<int> active_grid_dims;
      for (int p = 0; p < num_grid_dims; ++p) {
        bool used = false;
        for (auto const &imap : combo.input_maps) {
          if (imap[p] != -1) { used = true; break; }
        }
        if (!used) {
          for (auto const &omap : combo.output_maps) {
            if (omap[p] != -1) { used = true; break; }
          }
        }
        if (used) active_grid_dims.push_back(p);
      }

      int collapsed_num_grid_dims = (int)active_grid_dims.size();
      if (collapsed_num_grid_dims == 0) {
        // All grid dims unused — use 1 grid dim with all -1 maps
        collapsed_num_grid_dims = 1;
        active_grid_dims.push_back(0);
      }

      // Build collapsed maps: keep only active grid dim columns
      std::vector<std::vector<int>> collapsed_input_maps;
      for (auto const &imap : combo.input_maps) {
        std::vector<int> collapsed(collapsed_num_grid_dims);
        for (int i = 0; i < collapsed_num_grid_dims; ++i) {
          collapsed[i] = imap[active_grid_dims[i]];
        }
        collapsed_input_maps.push_back(collapsed);
      }
      std::vector<std::vector<int>> collapsed_output_maps;
      for (auto const &omap : combo.output_maps) {
        std::vector<int> collapsed(collapsed_num_grid_dims);
        for (int i = 0; i < collapsed_num_grid_dims; ++i) {
          collapsed[i] = omap[active_grid_dims[i]];
        }
        collapsed_output_maps.push_back(collapsed);
      }

      // Dedup: skip if we've already verified this collapsed map combo
      auto collapsed_key = std::make_tuple(
          kn_input_shapes, op_type_seq,
          collapsed_input_maps, combo.forloop_dims, collapsed_output_maps);
      {
        std::lock_guard<std::mutex> lock(seen_collapsed_mutex);
        if (!seen_collapsed.insert(collapsed_key).second) {
          continue;
        }
      }

      auto concrete_tb = rebuild_tb_with_concrete_maps(
          tb, collapsed_input_maps, combo.forloop_dims, collapsed_output_maps,
          collapsed_num_grid_dims);
      if (!concrete_tb.has_value()) {
        continue;
      }
      SymbolicKNGraph concrete_kn = rebuild_kn_with_tb(graph, op_idx, *concrete_tb);
      OutputMatch match = verify_symbolic_graph(concrete_kn);
      if (match.is_valid()) {
        results.push_back({concrete_kn, match});
      }
    }
    break; // Only handle the first KN_CUSTOMIZED_OP
  }

  return results;
}

} // namespace search
} // namespace mirage
