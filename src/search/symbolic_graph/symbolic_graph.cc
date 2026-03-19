#include "mirage/search/symbolic_graph/symbolic_graph.h"
#include "mirage/kernel/device_tensor.h"
#include "mirage/kernel/operator.h"
#include "mirage/layout.h"
#include "mirage/search/op_utils.h"
#include "mirage/search/symbolic_graph/op_args.h"
#include "mirage/search/symbolic_graph/symbolic_map.h"
#include "mirage/search/symbolic_graph/symbolic_tensor.h"
#include "mirage/search/symbolic_graph/tensor_dim_expr.h"
#include "mirage/type.h"
#include "mirage/utils/containers.h"

#include <iostream>
#include <numeric>
#include <optional>
#include <unordered_set>

namespace mirage {
namespace search {

SymbolicTBGraph::SymbolicTBGraph(tensor_dim_var_index_t dim_variable_index_base, int num_parallel_dims)
    : dim_variable_index_base(dim_variable_index_base),
      next_dim_variable_index(dim_variable_index_base) {
  assert(num_parallel_dims <= 3);
  for (int i = 0; i < num_parallel_dims; ++i) {
    grid_dim.push_back(std::make_shared<TensorDimVar>(next_dim_variable_index++));
  }
  block_dim.push_back(std::make_shared<TensorDimConst>(128));
  block_dim.push_back(std::make_shared<TensorDimConst>(1));
  block_dim.push_back(std::make_shared<TensorDimConst>(1));
  forloop_range = SymbolicTensorDim(std::make_shared<TensorDimVar>(next_dim_variable_index++));
}

bool SymbolicTBGraph::remove_last_operator() {
  if (operators.empty()) {
    return false;
  }
  operators.pop_back();
  for (size_t i = 0; i < output_indices.back().size(); i++) {
    tensors.pop_back();
  }
  input_indices.pop_back();
  output_indices.pop_back();
  return true;
}

threadblock::Graph *SymbolicTBGraph::to_threadblock_graph(
    DimVarAssignment const &assignment,
    std::vector<kernel::DTensor> const &inputs) const {
  std::vector<unsigned int> grid_dim_val_vec;
  for (size_t i = 0; i < grid_dim.size(); ++i) {
    grid_dim_val_vec.push_back(assignment.get_value(grid_dim[i]));
  }
  dim3 grid_dim_val = vec_to_dim3(pad_vector(grid_dim_val_vec, 3, 1u));
  dim3 block_dim_val(assignment.get_value(block_dim[0]),
                     assignment.get_value(block_dim[1]),
                     assignment.get_value(block_dim[2]));
  int forloop_range_val = assignment.get_value(forloop_range);
  int reduction_dimx = [&]() {
    std::unordered_set<int> reduction_dimx_candidates;
    for (size_t i = 0; i < this->operators.size(); ++i) {
      if (this->operators[i].op_type == type::TBOperatorType::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP) {
        std::shared_ptr<TBReductionOpArgs const> args = std::static_pointer_cast<TBReductionOpArgs const>(this->operators[i].args);
        SymbolicSTensor reduced_tensor = this->tensors[this->output_indices[i][0]];
        int reduction_dimx = assignment.get_value(reduced_tensor.dims[args->reduce_dim]);
        reduction_dimx_candidates.insert(reduction_dimx);
      }
    }
    if (reduction_dimx_candidates.empty()) {
      return 1;
    }
    if (reduction_dimx_candidates.size() == 1) {
      return *reduction_dimx_candidates.begin();
    }
    return -1;
  }();
  if (reduction_dimx == -1) {
    return nullptr;
  }
  threadblock::Graph *graph = new threadblock::Graph(
      grid_dim_val, block_dim_val, forloop_range_val, reduction_dimx);

  std::vector<threadblock::STensor> tensors_val;

  for (size_t i = 0; i < this->operators.size(); ++i) {
    threadblock::TBOperator *op = nullptr;

    if (this->operators[i].op_type == type::TBOperatorType::TB_INPUT_OP) {
      kernel::DTensor dtensor = inputs[i];
      TBInputOpArgs const *args = static_cast<TBInputOpArgs const *>(this->operators[i].args.get());
      int3 input_map = vec_to_int3(args->input_map);
      int forloop_dim = args->forloop_dim;
      op = graph->create_input_op(dtensor, input_map, forloop_dim, layout::SmemRowMajor, false);
    } else if (this->operators[i].op_type == type::TBOperatorType::TB_OUTPUT_OP) {
      std::vector<threadblock::STensor> input_tensors;
      for (int input_index : this->input_indices[i]) {
        input_tensors.push_back(tensors_val[input_index]);
      }
      TBOutputOpArgs const *args = static_cast<TBOutputOpArgs const *>(this->operators[i].args.get());
      int3 output_map = vec_to_int3(args->output_map);
      mirage::type::TBEpilogueType epilogue = args->epilogue;
      op = graph->create_output_op(input_tensors[0], output_map, -1, epilogue);
    } else {
      std::vector<threadblock::STensor> input_tensors;
      for (int input_index : this->input_indices[i]) {
        input_tensors.push_back(tensors_val[input_index]);
      }
      op = create_op(*graph, this->operators[i].op_type, input_tensors);
    }
    if (op == nullptr) {
      delete graph;
      return nullptr;
    }
    graph->operators.push_back(op);
    if (op->output_tensors.size() > 0) {
      tensors_val.push_back(op->output_tensors[0]);
    }
  }
  return graph;
} 

TensorDimConstraint SymbolicTBGraph::get_memory_usage_constraint() const {
  SymbolicTensorDim total_size = dim_expr_make_const(0);
  for (size_t i = 0; i < this->operators.size(); i++) {
    if (is_unary(this->operators[i].op_type) || this->operators[i].op_type == type::TBOperatorType::TB_FORLOOP_ACCUM_NO_RED_OP) {
      continue;
    }
    for (size_t j = 0; j < this->output_indices[i].size(); ++j) {
      total_size = total_size + get_tensor_size(this->tensors[this->output_indices[i][j]]);
    }
  }
  return total_size <= mirage::config::MAX_SMEM_SIZE;
}

bool SymbolicTBGraph::check_memory_usage() {
  assert(false && "TBD");
  return true;
}

// Recursively walk an expression tree; for every node of the form
// TensorDimDiv(const C, var v) where v->index == var_index, push C.
static void collect_divisor_constants(
    SymbolicTensorDim const &expr,
    tensor_dim_var_index_t var_index,
    std::vector<int> &out) {
  if (expr->is_div()) {
    auto d = std::static_pointer_cast<TensorDimDiv const>(expr);
    if (d->rhs->is_var() && d->lhs->is_const()) {
      auto var = std::static_pointer_cast<TensorDimVar const>(d->rhs);
      if (var->index == var_index) {
        auto c = std::static_pointer_cast<TensorDimConst const>(d->lhs);
        out.push_back(c->value);
      }
    }
    collect_divisor_constants(d->lhs, var_index, out);
    collect_divisor_constants(d->rhs, var_index, out);
  } else if (expr->is_mul()) {
    auto m = std::static_pointer_cast<TensorDimMul const>(expr);
    collect_divisor_constants(m->lhs, var_index, out);
    collect_divisor_constants(m->rhs, var_index, out);
  } else if (expr->is_add()) {
    auto a = std::static_pointer_cast<TensorDimAdd const>(expr);
    collect_divisor_constants(a->lhs, var_index, out);
    collect_divisor_constants(a->rhs, var_index, out);
  }
  // Var, Const, Ite, comparison nodes: nothing to collect.
}

// Recursively verify that every TensorDimDiv node in the expression
// evaluates without remainder under the given assignment.
static bool check_dim_divisibility(SymbolicTensorDim const &expr,
                                   DimVarAssignment const &assignment) {
  if (expr->is_div()) {
    auto d = std::static_pointer_cast<TensorDimDiv const>(expr);
    auto lv = d->lhs->maybe_get_value(assignment);
    auto rv = d->rhs->maybe_get_value(assignment);
    if (lv && rv && (*rv == 0 || *lv % *rv != 0)) return false;
    return check_dim_divisibility(d->lhs, assignment) &&
           check_dim_divisibility(d->rhs, assignment);
  }
  if (expr->is_mul()) {
    auto m = std::static_pointer_cast<TensorDimMul const>(expr);
    return check_dim_divisibility(m->lhs, assignment) &&
           check_dim_divisibility(m->rhs, assignment);
  }
  if (expr->is_add()) {
    auto a = std::static_pointer_cast<TensorDimAdd const>(expr);
    return check_dim_divisibility(a->lhs, assignment) &&
           check_dim_divisibility(a->rhs, assignment);
  }
  return true;
}

SymbolicTBGraph SymbolicTBGraph::with_updated_input_shapes(
    std::vector<SymbolicDTensor> const &new_input_dtensors) const {
  // Rebuild the TB graph by replaying operators with new input dtensors.
  // This correctly recomputes all downstream STensor dims.
  //
  // We use placement-style init to preserve the original grid_dim/forloop_range
  // symbolic variables (the constructor would create fresh ones).
  SymbolicTBGraph result(this->dim_variable_index_base,
                         static_cast<int>(this->grid_dim.size()));
  result.grid_dim = this->grid_dim;
  result.block_dim = this->block_dim;
  result.forloop_range = this->forloop_range;
  result.reduction_degree = this->reduction_degree;
  result.next_dim_variable_index = this->next_dim_variable_index;

  size_t input_idx = 0;
  for (size_t i = 0; i < this->operators.size(); ++i) {
    auto const &op = this->operators[i];
    if (op.op_type == type::TBOperatorType::TB_INPUT_OP) {
      TBInputOpArgs const *args =
          static_cast<TBInputOpArgs const *>(op.args.get());
      SymbolicDTensor dtensor = (input_idx < new_input_dtensors.size())
                                    ? new_input_dtensors[input_idx]
                                    : args->dtensor;
      result.add_input(dtensor, args->input_map, args->forloop_dim);
      ++input_idx;
    } else if (op.op_type == type::TBOperatorType::TB_OUTPUT_OP) {
      TBOutputOpArgs const *args =
          static_cast<TBOutputOpArgs const *>(op.args.get());
      int tb_input_index = this->input_indices[i][0];
      result.add_output(tb_input_index, args->output_map, args->epilogue);
    } else {
      result.add_operator(op.op_type, this->input_indices[i]);
    }
  }
  return result;
}

int SymbolicTBGraph::get_initial_value_for_var(
    tensor_dim_var_index_t var_index) const {
  std::vector<int> constants;
  for (auto const &tensor : tensors) {
    for (auto const &dim : tensor.dims) {
      collect_divisor_constants(dim, var_index, constants);
    }
  }
  if (constants.empty()) return 4;  // no divisibility constraint; safe default
  int g = constants[0];
  for (size_t i = 1; i < constants.size(); ++i) {
    g = std::gcd(g, constants[i]);
  }
  return g;  // largest value that divides all constants exactly
}

bool SymbolicTBGraph::is_valid_assignment(
    DimVarAssignment const &assignment) const {
  // (a) Divisibility: every C/v node must divide evenly.
  for (auto const &tensor : tensors) {
    for (auto const &dim : tensor.dims) {
      if (!check_dim_divisibility(dim, assignment)) return false;
    }
  }
  // (b) Smem: mirror get_memory_usage_constraint() logic but in bytes (fp16 = 2).
  // get_tensor_size() returns element count; multiply by 2 to get bytes.
  SymbolicTensorDim total_elems = dim_expr_make_const(0);
  for (size_t i = 0; i < operators.size(); ++i) {
    if (is_unary(operators[i].op_type) ||
        operators[i].op_type ==
            type::TBOperatorType::TB_FORLOOP_ACCUM_NO_RED_OP) {
      continue;
    }
    for (int idx : output_indices[i]) {
      total_elems = total_elems + get_tensor_size(tensors[idx]);
    }
  }
  auto elems = total_elems->maybe_get_value(assignment);
  if (!elems.has_value()) return false;          // unresolved variable
  return (*elems * 2) <= (int)mirage::config::MAX_SMEM_SIZE;  // fp16 = 2 bytes
}

bool SymbolicTBGraph::add_operator(type::TBOperatorType op_type,
                                   std::vector<int> input_indices) {
  for (size_t i = 1; i < input_indices.size(); i++) {
    if (tensors[input_indices[i]].after_accum !=
        tensors[input_indices[0]].after_accum) {
      return false;
    }
  }

  switch (op_type) {
    case type::TBOperatorType::TB_CONCAT_0_OP:
    case type::TBOperatorType::TB_CONCAT_1_OP:
    case type::TBOperatorType::TB_CONCAT_2_OP: {
      int concat_dim = (int)op_type - (int)type::TBOperatorType::TB_CONCAT_0_OP;
      assert(input_indices.size() == 2);
      SymbolicSTensor A = this->tensors[input_indices[0]];
      SymbolicSTensor B = this->tensors[input_indices[1]];
      if (A.dims.size() != B.dims.size()) {
        return false;
      }
      if (concat_dim > (int)A.dims.size()) {
        return false;
      }
      {
        for (size_t i = 0; i < A.dims.size(); i++) {
          if ((int)i != concat_dim && !A.dims[i]->symbolically_equivalent_to(B.dims[i])) {
            return false;
          }
        }
      }
      this->operators.push_back(
          SymbolicTBOp(op_type, std::make_shared<TBConcatOpArgs>(concat_dim)));
      {
        std::vector<SymbolicTensorDim> dim_templates;
        for (size_t i = 0; i < A.dims.size(); i++) {
          if ((int)i != concat_dim) {
            dim_templates.push_back(A.dims[i]);
          } else {
            dim_templates.push_back(A.dims[i] + B.dims[i]);
          }
        }
        SymbolicSTensor C(dim_templates, A.after_accum);
        this->tensors.push_back(C);
      }
      this->input_indices.push_back(input_indices);
      this->output_indices.push_back({(int)this->tensors.size() - 1});
      break;
    }
    case type::TBOperatorType::TB_EXP_OP:
    case type::TBOperatorType::TB_SQUARE_OP:
    case type::TBOperatorType::TB_SQRT_OP:
    case type::TBOperatorType::TB_SILU_OP:
    case type::TBOperatorType::TB_MUL_SCALAR_OP: {
      assert(input_indices.size() == 1);
      SymbolicSTensor A = this->tensors[input_indices[0]];
      this->operators.push_back(SymbolicTBOp(op_type));
      this->tensors.push_back(A);
      this->input_indices.push_back(input_indices);
      this->output_indices.push_back({(int)this->tensors.size() - 1});
      break;
    }
    case type::TBOperatorType::TB_ADD_OP:
    case type::TBOperatorType::TB_MUL_OP:
    case type::TBOperatorType::TB_DIV_OP: {
      assert(input_indices.size() == 2);
      // Allow mul(x, x) (e.g. rms*rms), but reject add(x,x) and div(x,x)
      if (input_indices[0] == input_indices[1] &&
          op_type != type::TBOperatorType::TB_MUL_OP) {
        return false;
      }
      SymbolicSTensor A = this->tensors[input_indices[0]];
      SymbolicSTensor B = this->tensors[input_indices[1]];
      if (A.dims.size() != B.dims.size()) {
        return false;
      }
      {
        for (size_t i = 0; i < A.dims.size(); i++) {
          if (A.dims[i]->is_one() || B.dims[i]->is_one()) {
            continue;
          }
          if (A.dims[i]->symbolically_equivalent_to(B.dims[i])) {
            continue;
          }
          return false;
        }
      }
      this->operators.push_back(SymbolicTBOp(op_type));
      this->tensors.push_back(A);
      this->input_indices.push_back(input_indices);
      this->output_indices.push_back({(int)this->tensors.size() - 1});
      break;
    }
    case type::TBOperatorType::TB_FORLOOP_ACCUM_NO_RED_OP:
    case type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_SUM_OP:
    case type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_MEAN_OP:
    case type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_RMS_OP:
    case type::TBOperatorType::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP: {
      assert(input_indices.size() == 1);
      SymbolicSTensor A = this->tensors[input_indices[0]];
      if (A.after_accum) {
        return false;
      }
      {
        if (A.dims[A.dims.size() - 1]->is_one()) {
          return false;
        }
      }
      this->operators.push_back(SymbolicTBOp(op_type));
      {
        std::vector<SymbolicTensorDim> dim_templates;
        for (size_t i = 0; i < A.dims.size(); i++) {
          dim_templates.push_back(A.dims[i]);
        }
        if (op_type == type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_SUM_OP ||
            op_type == type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_MEAN_OP ||
            op_type == type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_RMS_OP) {
          dim_templates[dim_templates.size() - 1] = dim_expr_make_const(1);
        } else if (op_type ==
                   type::TBOperatorType::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP) {
          dim_templates[dim_templates.size() - 1] = dim_templates[dim_templates.size() - 1] / this->reduction_degree;
        }
        SymbolicSTensor B(dim_templates, true);
        this->tensors.push_back(B);
      }
      this->input_indices.push_back(input_indices);
      this->output_indices.push_back({(int)this->tensors.size() - 1});
      break;
    }
    case type::TBOperatorType::TB_MATMUL_OP: {
      assert(input_indices.size() == 2);
      SymbolicSTensor A = this->tensors[input_indices[0]];
      SymbolicSTensor B = this->tensors[input_indices[1]];
      if (A.after_accum) {
        return false;
      }
      if (A.dims.size() != B.dims.size()) {
        return false;
      }
      {
        for (size_t i = 0; i < A.dims.size() - 2; i++) {
          if (A.dims[i]->is_one() || B.dims[i]->is_one()) {
            continue;
          }
          if (A.dims[i]->symbolically_equivalent_to(B.dims[i])) {
            continue;
          }
          return false;
        }
        if (!A.dims[A.dims.size() - 1]->symbolically_equivalent_to(B.dims[B.dims.size() - 2])) {
          return false;
        }
      }
      this->operators.push_back(SymbolicTBOp(op_type));
      {
        std::vector<SymbolicTensorDim> dim_templates = A.dims;
        dim_templates[dim_templates.size() - 1] = B.dims[B.dims.size() - 1];
        SymbolicSTensor C(dim_templates, A.after_accum);
        this->tensors.push_back(C);
      }
      this->input_indices.push_back(input_indices);
      this->output_indices.push_back({(int)this->tensors.size() - 1});
      break;
    }
    case type::TBOperatorType::TB_REDUCTION_0_OP:
    case type::TBOperatorType::TB_REDUCTION_1_OP:
    case type::TBOperatorType::TB_REDUCTION_2_OP: {
      int reduction_dim =
          (int)op_type - (int)type::TBOperatorType::TB_REDUCTION_0_OP;
      assert(input_indices.size() == 1);
      SymbolicSTensor A = this->tensors[input_indices[0]];
      if ((int)A.dims.size() <= reduction_dim) {
        return false;
      }
      this->operators.push_back(SymbolicTBOp(
          op_type, std::make_shared<TBReductionOpArgs>(reduction_dim, dim_expr_make_const(1))));
      {
        std::vector<SymbolicTensorDim> dim_templates = A.dims;
        dim_templates[reduction_dim] = dim_expr_make_const(1);
        SymbolicSTensor B(dim_templates, A.after_accum);
        this->tensors.push_back(B);
      }
      this->input_indices.push_back(input_indices);
      this->output_indices.push_back({(int)this->tensors.size() - 1});
      break;
    }
    case type::TBOperatorType::TB_REDUCTION_0_TO_DIMX_OP:
    case type::TBOperatorType::TB_REDUCTION_1_TO_DIMX_OP:
    case type::TBOperatorType::TB_REDUCTION_2_TO_DIMX_OP: {
      int reduction_dim =
          (int)op_type - (int)type::TBOperatorType::TB_REDUCTION_0_TO_DIMX_OP;
      assert(input_indices.size() == 1);
      SymbolicSTensor A = this->tensors[input_indices[0]];
      if ((int)A.dims.size() <= reduction_dim) {
        return false;
      }
      this->operators.push_back(
          SymbolicTBOp(op_type,
                       std::make_shared<TBReductionOpArgs>(
                           reduction_dim, this->reduction_degree)));
      {
        std::vector<SymbolicTensorDim> dim_templates = A.dims;
        dim_templates[reduction_dim] = dim_templates[reduction_dim] / this->reduction_degree;
        SymbolicSTensor B(dim_templates, A.after_accum);
        this->tensors.push_back(B);
      }
      this->input_indices.push_back(input_indices);
      this->output_indices.push_back({(int)this->tensors.size() - 1});
      break;
    }
    case type::TBOperatorType::TB_RMS_NORM_OP: {
      assert(input_indices.size() == 1);
      SymbolicSTensor A = this->tensors[input_indices[0]];
      this->operators.push_back(SymbolicTBOp(op_type));
      this->tensors.push_back(A);
      this->input_indices.push_back(input_indices);
      this->output_indices.push_back({(int)this->tensors.size() - 1});
      break;
    }
    default: {
      return false;
    }
  }
  return true;
}

bool SymbolicTBGraph::add_input(SymbolicDTensor dtensor, std::vector<int> imap, int forloop_dim) {
  // auto get_data_dims = [](SymbolicDTensor const &dtensor) {
  //   std::vector<size_t> data_dims;
  //   for (size_t i = 0; i < dtensor.dims.size(); ++i) {
  //     data_dims.push_back(i);
  //   }
  //   return data_dims;
  // };
  // auto get_parallel_dims = [](std::vector<SymbolicTensorDim> const &grid_dim, SymbolicTensorDim const &forloop_range) {
  //   std::vector<SymbolicTensorDim> parallel_dims = grid_dim;
  //   parallel_dims.push_back(forloop_range);
  //   return parallel_dims;
  // };
  // SymbolicIMap imap(
  //   get_parallel_dims(grid_dim, forloop_range),
  //   get_data_dims(dtensor),
  //   next_dim_variable_index
  // );

  std::shared_ptr<OpArgs const> args =
      std::make_shared<TBInputOpArgs const>(dtensor, imap, forloop_dim);
  SymbolicTBOp op(type::TBOperatorType::TB_INPUT_OP, args);

  // auto compute_symbolic_stensor = [&](SymbolicDTensor const &dtensor,
  //                                    SymbolicIMap const &imap) {
  //   std::vector<SymbolicTensorDim> dim_templates = dtensor.dims;
  //   assert(dim_templates.size() == imap.data_dims.size());
  //   for (size_t i = 0; i < dim_templates.size(); ++i) {
  //     SymbolicTensorDim divisor = dim_expr_make_ite(imap.mat.at({grid_dim[0], i}) == 1, grid_dim[0], dim_expr_make_const(1));
  //     for (size_t j = 1; j < grid_dim.size(); ++j) {
  //       divisor = divisor * dim_expr_make_ite(imap.mat.at({grid_dim[j], i}) == 1, grid_dim[j], dim_expr_make_const(1));
  //     }
  //     divisor = divisor * dim_expr_make_ite(imap.mat.at({forloop_range, i}) == 1, forloop_range, dim_expr_make_const(1));
  //     dim_templates[i] = dim_templates[i] / divisor;
  //   }
  //   return SymbolicSTensor(dim_templates, false);
  // };
  auto compute_symbolic_stensor = [&]() {
    std::vector<SymbolicTensorDim> dim_templates = dtensor.dims;
    for (size_t i = 0; i < imap.size(); ++i) {
      if (imap[i] != -1) {
        dim_templates[imap[i]] = dim_templates[imap[i]] / grid_dim[i];
      }
    }
    if (forloop_dim != -1) {
      dim_templates[forloop_dim] = dim_templates[forloop_dim] / forloop_range;
    }
    return SymbolicSTensor(dim_templates, false);
  };

  SymbolicSTensor tensor = compute_symbolic_stensor();

  operators.push_back(op);
  tensors.push_back(tensor);
  input_indices.push_back({});
  output_indices.push_back({(int)tensors.size() - 1});

  return true;
}

bool SymbolicTBGraph::add_output(int input_index, std::vector<int> omap, type::TBEpilogueType epilogue_type) {
  // auto get_data_dims = [](SymbolicSTensor const &stensor) {
  //   std::vector<size_t> data_dims;
  //   for (size_t i = 0; i < stensor.dims.size(); ++i) {
  //     data_dims.push_back(i);
  //   }
  //   return data_dims;
  // };
  // auto get_parallel_dims = [](std::vector<SymbolicTensorDim> const &grid_dim) {
  //   return grid_dim;
  // };
  // SymbolicOmap omap(
  //   get_parallel_dims(grid_dim),
  //   get_data_dims(this->tensors[input_index]),
  //   next_dim_variable_index
  // );

  // auto compute_symbolic_dtensor = [&](SymbolicSTensor const &stensor,
  //                                    SymbolicOmap const &omap) {
  //   std::vector<SymbolicTensorDim> dim_templates = stensor.dims;
  //   assert(dim_templates.size() == omap.data_dims.size());
  //   for (size_t i = 0; i < dim_templates.size(); ++i) {
  //     SymbolicTensorDim multiplier = dim_expr_make_ite(omap.mat.at({grid_dim[0], i}) == 1, grid_dim[0], dim_expr_make_const(1));
  //     for (size_t j = 1; j < grid_dim.size(); ++j) {
  //       multiplier = multiplier * dim_expr_make_ite(omap.mat.at({grid_dim[j], i}) == 1, grid_dim[j], dim_expr_make_const(1));
  //     }
  //     dim_templates[i] = dim_templates[i] * multiplier;
  //   }
  //   return SymbolicDTensor(dim_templates);
  // };
  auto compute_symbolic_dtensor = [&]() {
    std::vector<SymbolicTensorDim> dim_templates = this->tensors[input_index].dims;
    for (size_t i = 0; i < omap.size(); ++i) {
      if (omap[i] != -1) {
        dim_templates[omap[i]] = dim_templates[omap[i]] * grid_dim[i];
      }
    }
    return SymbolicDTensor(dim_templates);
  };

  SymbolicDTensor dtensor = compute_symbolic_dtensor();
  std::shared_ptr<OpArgs const> args =
      std::make_shared<TBOutputOpArgs const>(dtensor, omap, epilogue_type);
  SymbolicTBOp op(type::TBOperatorType::TB_OUTPUT_OP, args);
  operators.push_back(op);
  input_indices.push_back({input_index});
  output_indices.push_back({});
  return true;
}

mirage::kernel::Graph *SymbolicKNGraph::to_kernel_graph(
    DimVarAssignment const &assignment) const {
  kernel::Graph *graph = new kernel::Graph();
  std::vector<kernel::DTensor> tensors_val;
  for (size_t i = 0; i < this->operators.size(); ++i) {
    std::vector<kernel::DTensor> input_tensors;
    for (int input_index : this->input_indices[i]) {
      input_tensors.push_back(tensors_val[input_index]);
    }
    kernel::KNOperator *op = nullptr;
    if (this->operators[i].op_type == type::KNOperatorType::KN_INPUT_OP) {
      std::shared_ptr<KNInputOpArgs const> args =
          std::static_pointer_cast<KNInputOpArgs const>(
              this->operators[i].args);
      op = graph->create_input_op(args->input_dims, args->input_strides, args->data_type, args->layout);
    } else if (this->operators[i].op_type == type::KNOperatorType::KN_CUSTOMIZED_OP) {
      std::shared_ptr<KNCustomizedOpArgs const> args =
          std::static_pointer_cast<KNCustomizedOpArgs const>(
              this->operators[i].args);
      threadblock::Graph *tb_graph =
          args->tb_graph_template.to_threadblock_graph(assignment,
                                                       input_tensors);
      if (tb_graph == nullptr) {
        delete graph;
        return nullptr;
      }
      op = graph->create_customized_op(input_tensors, *tb_graph);
    } else {
      op = create_op(*graph, this->operators[i].op_type, input_tensors);
    }
    if (op == nullptr) {
      delete graph;
      return nullptr;
    }
    graph->operators.push_back(op);
    for (DTensor const &output_tensor : op->output_tensors) {
      tensors_val.push_back(output_tensor);
    }
  }
  // Mark output tensors: any tensor not consumed as an input by a later op.
  std::unordered_set<int> consumed;
  for (auto const &idx_list : this->input_indices) {
    for (int idx : idx_list) {
      consumed.insert(idx);
    }
  }
  for (size_t i = 0; i < tensors_val.size(); ++i) {
    if (consumed.find((int)i) == consumed.end()) {
      graph->mark_output(tensors_val[i]);
    }
  }
  return graph;
}

bool SymbolicKNGraph::remove_last_operator() {
  if (operators.empty()) {
    return false;
  }
  if (operators.back().op_type == type::KNOperatorType::KN_CUSTOMIZED_OP) {
    SymbolicKNOp op = operators.back();
    std::shared_ptr<KNCustomizedOpArgs const> args =
        std::static_pointer_cast<KNCustomizedOpArgs const>(op.args);
    next_dim_variable_index = args->tb_graph_template.dim_variable_index_base;
  }
  operators.pop_back();
  for (int _ : output_indices.back()) {
    tensors.pop_back();
  }
  input_indices.pop_back();
  output_indices.pop_back();
  return true;
}

bool SymbolicKNGraph::add_operator(type::KNOperatorType op_type,
                                   std::vector<int> input_indices) {
  switch (op_type) {
    case type::KNOperatorType::KN_ADD_OP:
    case type::KNOperatorType::KN_MUL_OP:
    case type::KNOperatorType::KN_DIV_OP: {
      assert(input_indices.size() == 2);
      SymbolicDTensor A = this->tensors[input_indices[0]];
      SymbolicDTensor B = this->tensors[input_indices[1]];
      if (A.dims.size() != B.dims.size()) {
        return false;
      }
      {
        for (size_t i = 0; i < A.dims.size(); i++) {
          if (A.dims[i]->is_one() || B.dims[i]->is_one()) {
            continue;
          }
          if (A.dims[i]->symbolically_equivalent_to(B.dims[i])) {
            continue;
          }
          return false;
        }
      }
      this->operators.push_back(SymbolicKNOp(op_type));
      this->tensors.push_back(A);
      this->input_indices.push_back(input_indices);
      this->output_indices.push_back({(int)this->tensors.size() - 1});
      break;
    }
    case type::KNOperatorType::KN_EXP_OP:
    case type::KNOperatorType::KN_SQUARE_OP:
    case type::KNOperatorType::KN_SQRT_OP:
    case type::KNOperatorType::KN_SILU_OP: {
      assert(input_indices.size() == 1);
      SymbolicDTensor A = this->tensors[input_indices[0]];
      this->operators.push_back(SymbolicKNOp(op_type));
      this->tensors.push_back(A);
      this->input_indices.push_back(input_indices);
      this->output_indices.push_back({(int)this->tensors.size() - 1});
      break;
    }
    case type::KNOperatorType::KN_MATMUL_OP: {
      assert(input_indices.size() == 2);
      SymbolicDTensor A = this->tensors[input_indices[0]];
      SymbolicDTensor B = this->tensors[input_indices[1]];
      if (A.dims.size() != B.dims.size()) {
        return false;
      }
      {
        for (size_t i = 0; i < A.dims.size() - 2; i++) {
          if (A.dims[i]->symbolically_equivalent_to(B.dims[i])) {
            continue;
          }
          return false;
        }
        if (!A.dims[A.dims.size() - 1]->symbolically_equivalent_to(B.dims[B.dims.size() - 2])) {
          return false;
        }
      }
      this->operators.push_back(SymbolicKNOp(op_type));
      {
        std::vector<SymbolicTensorDim> dim_templates;
        for (size_t i = 0; i < A.dims.size(); i++) {
          dim_templates.push_back(A.dims[i]);
        }
        dim_templates[dim_templates.size() - 1] = B.dims[B.dims.size() - 1];
        SymbolicDTensor C(dim_templates);
        this->tensors.push_back(C);
      }
      this->input_indices.push_back(input_indices);
      this->output_indices.push_back({(int)this->tensors.size() - 1});
      break;
    }
    case type::KNOperatorType::KN_REDUCTION_0_OP:
    case type::KNOperatorType::KN_REDUCTION_1_OP:
    case type::KNOperatorType::KN_REDUCTION_2_OP: {
      int reduction_dim =
          (int)op_type - (int)type::KNOperatorType::KN_REDUCTION_0_OP;
      int reduction_dim_size = 1;
      assert(input_indices.size() == 1);
      SymbolicDTensor A = this->tensors[input_indices[0]];
      if ((int)A.dims.size() <= reduction_dim) {
        return false;
      }
      this->operators.push_back(
          SymbolicKNOp(op_type,
                       std::make_shared<KNReductionOpArgs>(
                           reduction_dim, reduction_dim_size)));
      {
        std::vector<SymbolicTensorDim> dim_templates = A.dims;
        dim_templates[reduction_dim] = dim_expr_make_const(reduction_dim_size);
        SymbolicDTensor B(dim_templates);
        this->tensors.push_back(B);
      }
      this->input_indices.push_back(input_indices);
      this->output_indices.push_back({(int)this->tensors.size() - 1});
      break;
    }
    default: {
      return false;
    }
  }
  return true;
}

bool SymbolicKNGraph::add_customized_operator(SymbolicTBGraph tb_graph,
                                              std::vector<int> input_indices) {
  this->operators.push_back(
      SymbolicKNOp(type::KNOperatorType::KN_CUSTOMIZED_OP,
                   std::make_shared<KNCustomizedOpArgs>(tb_graph)));
  std::vector<int> output_indices;
  for (auto const &op : tb_graph.operators) {
    if (op.op_type == type::TBOperatorType::TB_OUTPUT_OP) {
      std::shared_ptr<TBOutputOpArgs const> args =
          std::static_pointer_cast<TBOutputOpArgs const>(op.args);
      output_indices.push_back((int)this->tensors.size());
      this->tensors.push_back(args->dtensor);
    }
  }
  this->input_indices.push_back(input_indices);
  this->output_indices.push_back(output_indices);
  this->next_dim_variable_index = tb_graph.next_dim_variable_index;
  return true;
}

bool SymbolicKNGraph::add_input(std::vector<int> input_dims,
                                std::vector<size_t> input_strides,
                                mirage::type::DataType data_type,
                                mirage::layout::DmemLayout layout,
                                int3 input_map) {
  this->operators.push_back(
      SymbolicKNOp(type::KNOperatorType::KN_INPUT_OP,
                   std::make_shared<KNInputOpArgs>(input_dims, input_strides, data_type, layout, input_map)));
  {
    std::vector<SymbolicTensorDim> dim_templates;
    for (size_t i = 0; i < input_dims.size(); i++) {
      dim_templates.push_back(
          dim_expr_make_const(input_dims[i]));
    }
    SymbolicDTensor tensor(dim_templates);
    this->tensors.push_back(tensor);
  }
  this->input_indices.push_back({});
  this->output_indices.push_back({(int)this->tensors.size() - 1});
  return true;
}

bool SymbolicKNGraph::add_output(int input_index,
                                 std::vector<size_t> output_strides,
                                 int3 output_map) {
  this->operators.push_back(SymbolicKNOp(
      type::KNOperatorType::KN_OUTPUT_OP,
      std::make_shared<KNOutputOpArgs>(output_strides, output_map)));
  this->input_indices.push_back({input_index});
  this->output_indices.push_back({});
  return true;
}

SymbolicTBGraph::operator json() const {
  std::vector<json> grid_dim_json;
  for (auto const &dim : grid_dim) {
    grid_dim_json.push_back(json(*dim));
  }
  std::vector<json> block_dim_json;
  for (auto const &dim : block_dim) {
    block_dim_json.push_back(json(*dim));
  }
  json reduction_degree_json;
  if (reduction_degree) {
    reduction_degree_json = json(*reduction_degree);
  } else {
    reduction_degree_json = json(nullptr);
  }
  return json{
      {"grid_dim", grid_dim_json},
      {"block_dim", block_dim_json},
      {"forloop_range", *forloop_range},
      {"reduction_degree", reduction_degree_json},
      {"operators", operators},
      {"tensors", tensors},
      {"input_indices", input_indices},
      {"output_indices", output_indices},
      {"num_operators", operators.size()},
  };
}

SymbolicKNGraph::operator json() const {
  return json{
      {"operators", operators},
      {"tensors", tensors},
      {"input_indices", input_indices},
      {"output_indices", output_indices},
  };
}

void from_json(json const &j, SymbolicTBGraph &symbolic_tb_graph) {
  symbolic_tb_graph.grid_dim.clear();
  for (auto const &jd : j.at("grid_dim")) {
    SymbolicTensorDim dim;
    from_json(jd, dim);
    symbolic_tb_graph.grid_dim.push_back(dim);
  }
  symbolic_tb_graph.block_dim.clear();
  for (auto const &jd : j.at("block_dim")) {
    SymbolicTensorDim dim;
    from_json(jd, dim);
    symbolic_tb_graph.block_dim.push_back(dim);
  }
  from_json(j.at("forloop_range"), symbolic_tb_graph.forloop_range);
  if (j.at("reduction_degree").is_null()) {
    symbolic_tb_graph.reduction_degree = nullptr;
  } else {
    from_json(j.at("reduction_degree"), symbolic_tb_graph.reduction_degree);
  }
  symbolic_tb_graph.operators.clear();
  for (auto const &jop : j.at("operators")) {
    SymbolicTBOp op(type::TBOperatorType::TB_UNKOWN, nullptr);
    from_json(jop, op);
    symbolic_tb_graph.operators.push_back(op);
  }
  symbolic_tb_graph.tensors.clear();
  for (auto const &jt : j.at("tensors")) {
    SymbolicSTensor t(std::vector<SymbolicTensorDim>{}, false);
    from_json(jt, t);
    symbolic_tb_graph.tensors.push_back(t);
  }
  symbolic_tb_graph.input_indices = j.at("input_indices").get<std::vector<std::vector<int>>>();
  symbolic_tb_graph.output_indices = j.at("output_indices").get<std::vector<std::vector<int>>>();
}

void from_json(json const &j, SymbolicKNGraph &symbolic_kn_graph) {
  symbolic_kn_graph.operators.clear();
  for (auto const &jop : j.at("operators")) {
    SymbolicKNOp op(type::KNOperatorType::KN_UNKOWN, nullptr);
    from_json(jop, op);
    symbolic_kn_graph.operators.push_back(op);
  }
  symbolic_kn_graph.tensors.clear();
  for (auto const &jt : j.at("tensors")) {
    SymbolicDTensor t(std::vector<SymbolicTensorDim>{});
    from_json(jt, t);
    symbolic_kn_graph.tensors.push_back(t);
  }
  symbolic_kn_graph.input_indices = j.at("input_indices").get<std::vector<std::vector<int>>>();
  symbolic_kn_graph.output_indices = j.at("output_indices").get<std::vector<std::vector<int>>>();
}

namespace {

std::vector<size_t> default_strides_from_dims(std::vector<int> const &dims) {
  std::vector<size_t> strides(dims.size());
  size_t stride = 1;
  for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= dims[i];
  }
  return strides;
}

// If all dims are const, return their values; otherwise nullopt.
std::optional<std::vector<int>> get_concrete_dims(SymbolicDTensor const &t) {
  std::vector<int> dims;
  for (auto const &d : t.dims) {
    if (!d->is_const()) {
      return std::nullopt;
    }
    dims.push_back(
        std::static_pointer_cast<TensorDimConst const>(d)->value);
  }
  return dims;
}

} // namespace

SymbolicKNGraph construct_graph_with_different_input_shapes(
    SymbolicKNGraph const &ref_graph,
    std::vector<std::vector<int>> const &input_shapes) {
  SymbolicKNGraph result;
  result.next_dim_variable_index = ref_graph.next_dim_variable_index;

  int input_op_idx = 0;
  for (size_t i = 0; i < ref_graph.operators.size(); ++i) {
    SymbolicKNOp const &ref_op = ref_graph.operators[i];
    std::vector<int> const &ref_input_idx = ref_graph.input_indices[i];

    switch (ref_op.op_type) {
      case type::KNOperatorType::KN_INPUT_OP: {
        if (input_op_idx >= static_cast<int>(input_shapes.size())) {
          return SymbolicKNGraph();
        }
        std::shared_ptr<KNInputOpArgs const> args =
            std::static_pointer_cast<KNInputOpArgs const>(ref_op.args);
        std::vector<int> new_dims = input_shapes[input_op_idx++];
        std::vector<size_t> new_strides = default_strides_from_dims(new_dims);
        if (!result.add_input(new_dims, new_strides, args->data_type,
                             args->layout, args->input_map)) {
          return SymbolicKNGraph();
        }
        break;
      }
      case type::KNOperatorType::KN_OUTPUT_OP: {
        std::shared_ptr<KNOutputOpArgs const> args =
            std::static_pointer_cast<KNOutputOpArgs const>(ref_op.args);
        int input_index = ref_input_idx[0];
        std::vector<size_t> output_strides;
        if (input_index < static_cast<int>(result.tensors.size())) {
          auto concrete = get_concrete_dims(result.tensors[input_index]);
          if (concrete) {
            output_strides = default_strides_from_dims(*concrete);
          } else {
            output_strides = args->output_strides;
          }
        } else {
          output_strides = args->output_strides;
        }
        int3 output_map = args->output_map;
        if (!result.add_output(input_index, output_strides, output_map)) {
          return SymbolicKNGraph();
        }
        break;
      }
      case type::KNOperatorType::KN_CUSTOMIZED_OP: {
        std::shared_ptr<KNCustomizedOpArgs const> args =
            std::static_pointer_cast<KNCustomizedOpArgs const>(ref_op.args);
        // Build new dtensors from the KN-level input shapes for this op.
        std::vector<SymbolicDTensor> new_tb_dtensors;
        for (int kn_idx : ref_input_idx) {
          if (kn_idx < static_cast<int>(result.tensors.size())) {
            new_tb_dtensors.push_back(result.tensors[kn_idx]);
          }
        }
        // Rebuild the TB graph template with updated input shapes.
        // This recomputes all downstream STensor dims correctly.
        SymbolicTBGraph updated_tb =
            args->tb_graph_template.with_updated_input_shapes(new_tb_dtensors);
        if (!result.add_customized_operator(updated_tb, ref_input_idx)) {
          return SymbolicKNGraph();
        }
        break;
      }
      default:
        if (!result.add_operator(ref_op.op_type, ref_input_idx)) {
          return SymbolicKNGraph();
        }
        break;
    }
  }

  return result;
}

} // namespace search
} // namespace mirage
