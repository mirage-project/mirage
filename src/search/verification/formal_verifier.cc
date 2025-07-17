#include "mirage/search/verification/formal_verifier.h"
#include "mirage/search/op_utils.h"
#include <iostream>

namespace mirage {
namespace search {

std::mutex FormalVerifier::formal_verifier_mutex;

FormalVerifier::FormalVerifier(kernel::Graph const &input_graph) {
  input_exprs = get_concrete_exprs(input_graph, true, all_dims);
}

OutputMatch FormalVerifier::verify(kernel::Graph const &graph) {
  std::lock_guard<std::mutex> lock(formal_verifier_mutex);

  std::vector<std::string> graph_exprs =
      get_concrete_exprs(graph, false, all_dims);
  assert(input_exprs.size() == graph_exprs.size());

  auto verify_with_match = [&](OutputMatch const &match) {
    for (size_t i = 0; i < match.size(); i++) {
      bool is_equiv =
          check_equiv(input_exprs[i].c_str(), graph_exprs[match[i]].c_str());
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
                       bool with_output_ops,
                       std::unordered_set<std::string> &all_dims) {
  std::unordered_map<type::GuidType, std::string> tensor_exprs;

  std::string data_dim0 = "data_dim0";
  std::string data_dim1 = "data_dim1";
  std::string data_dim2 = "data_dim2";

  all_dims.insert(data_dim0);
  all_dims.insert(data_dim1);
  all_dims.insert(data_dim2);

  std::string data_dim[3] = {data_dim0, data_dim1, data_dim2};

  int custom_kernel_id = 0;
  int redtox_id = 0;

  auto calc_stensor_exprs = [&](threadblock::Graph const &graph) {
    std::string dx = "dx" + std::to_string(custom_kernel_id);
    std::string dy = "dy" + std::to_string(custom_kernel_id);
    std::string dz = "dz" + std::to_string(custom_kernel_id);
    std::string df = "df" + std::to_string(custom_kernel_id);
    all_dims.insert(dx);
    all_dims.insert(dy);
    all_dims.insert(dz);
    all_dims.insert(df);
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
          all_dims.insert(reddim);
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

} // namespace search
} // namespace mirage
