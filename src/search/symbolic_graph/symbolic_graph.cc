#include "mirage/search/symbolic_graph/symbolic_graph.h"
#include "mirage/search/op_utils.h"
#include "mirage/search/symbolic_graph/op_args.h"
#include "mirage/search/symbolic_graph/tensor_dim_expr.h"
#include "mirage/type.h"

#include <iostream>

namespace mirage {
namespace search {

SymbolicTBGraph::SymbolicTBGraph(tensor_dim_var_index_t dim_variable_index_base)
    : dim_variable_index_base(dim_variable_index_base),
      next_dim_variable_index(dim_variable_index_base),
      grid_dim({std::make_shared<TensorDimVar>(next_dim_variable_index++),
                    std::make_shared<TensorDimVar>(next_dim_variable_index++),
                    std::make_shared<TensorDimVar>(next_dim_variable_index++),
                std::make_shared<TensorDimVar>(next_dim_variable_index++),
                std::make_shared<TensorDimVar>(
                    next_dim_variable_index++)}),
      block_dim({std::make_shared<TensorDimConst>(128),
                 std::make_shared<TensorDimConst>(1),
                 std::make_shared<TensorDimConst>(1),
                 std::make_shared<TensorDimConst>(1)}),
      forloop_range(SymbolicTensorDim(
          std::make_shared<TensorDimVar>(next_dim_variable_index++))) {
  assert(conds.add_constraint(grid_dim[0] >= 1));
  assert(conds.add_constraint(grid_dim[1] >= 1));
  assert(conds.add_constraint(grid_dim[2] >= 1));
  assert(conds.add_constraint(forloop_range >= 1));
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
  assert(conds.revert());
  return true;
}

threadblock::Graph *SymbolicTBGraph::to_threadblock_graph(
    DimVarAssignments const &assignments,
    std::vector<kernel::DTensor> const &inputs) const {
  dim3 grid_dim_val(assignments.get_value(grid_dim[0]),
                    assignments.get_value(grid_dim[1]),
                    assignments.get_value(grid_dim[2]));
  dim3 block_dim_val(assignments.get_value(block_dim[0]),
                     assignments.get_value(block_dim[1]),
                     assignments.get_value(block_dim[2]));
  int forloop_range_val = assignments.get_value(forloop_range);
  threadblock::Graph *graph = new threadblock::Graph(
      grid_dim_val, block_dim_val, forloop_range_val, reduction_dimx);

  std::vector<threadblock::STensor> tensors_val;

  for (size_t i = 0; i < this->operators.size(); ++i) {
    std::vector<threadblock::STensor> input_tensors;
    for (int input_index : this->input_indices[i]) {
      input_tensors.push_back(tensors_val[input_index]);
    }
    threadblock::TBOperator *op =
        create_op(*graph, this->operators[i].op_type, input_tensors);
    if (op == nullptr) {
      delete graph;
      return nullptr;
    }
    graph->operators.push_back(op);
    tensors_val.push_back(op->output_tensors[0]);
  }
  return graph;
}

TensorDimConstraint SymbolicTBGraph::get_memory_usage_constraint() const {
  SymbolicTensorDim total_size = dim_expr_make_const(0);
  for (size_t i = 0; i < this->operators.size(); i++) {
    if (is_unary(this->operators[i].op_type) || this->operators[i].op_type == type::TBOperatorType::TB_FORLOOP_ACCUM_NO_RED_OP) {
      continue;
    }
  }
  return total_size <= mirage::config::MAX_SMEM_SIZE;
}

bool SymbolicTBGraph::check_memory_usage() {
  bool valid = this->conds.add_constraint(this->get_memory_usage_constraint());
  this->conds.revert();
  return valid;
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
        std::unordered_set<TensorDimConstraint> constraints;        
        for (size_t i = 0; i < A.dims.size(); i++) {
          if ((int)i != concat_dim) {
            std::shared_ptr<TensorDimExpr const> dim_expr = dim_expr_make_eq(A.dims[i], B.dims[i]);
            constraints.insert(dim_expr);
          }
        }
        if (!conds.add_constraints(constraints)) {
          return false;
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
      conds.add_constraints({});
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
      SymbolicSTensor A = this->tensors[input_indices[0]];
      SymbolicSTensor B = this->tensors[input_indices[1]];
      if (A.dims.size() != B.dims.size()) {
        return false;
      }
      {
        std::unordered_set<TensorDimConstraint> constraints;
        for (size_t i = 0; i < A.dims.size(); i++) {
          constraints.insert(dim_expr_make_disj(
            {A.dims[i] == B.dims[i], A.dims[i] == 1, B.dims[i] == 1}
          ));
        }
        if (!conds.add_constraints(constraints)) {
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
      conds.add_constraints({});
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
          dim_templates[dim_templates.size() - 1] = 
              dim_expr_make_const(this->reduction_dimx);
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
      if (A.dims.size() != B.dims.size()) {
        return false;
      }
      {
        std::unordered_set<TensorDimConstraint> constraints;
        for (size_t i = 0; i < A.dims.size() - 2; i++) {
          constraints.insert(A.dims[i] == 1);
          constraints.insert(B.dims[i] == 1);
        }
        constraints.insert(A.dims[A.dims.size() - 1] == B.dims[B.dims.size() - 2]);
        if (!conds.add_constraints(constraints)) {
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
      conds.add_constraints({});
      this->operators.push_back(SymbolicTBOp(
          op_type, std::make_shared<TBReductionOpArgs>(reduction_dim, 1)));
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
      conds.add_constraints({});
      this->operators.push_back(
          SymbolicTBOp(op_type,
                       std::make_shared<TBReductionOpArgs>(
                           reduction_dim, this->reduction_dimx)));
      {
        std::vector<SymbolicTensorDim> dim_templates = A.dims;
        dim_templates[reduction_dim] = dim_expr_make_const(this->reduction_dimx);
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
      conds.add_constraints({});
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
  // if (!this->check_memory_usage()) {
  //   this->remove_last_operator();
  //   return false;
  // }
  return true;
}

bool SymbolicTBGraph::add_input(SymbolicDTensor dtensor) {
  auto get_data_dims = [](SymbolicDTensor const &dtensor) {
    std::vector<size_t> data_dims;
    for (size_t i = 0; i < dtensor.dims.size(); ++i) {
      data_dims.push_back(i);
    }
    return data_dims;
  };
  SymbolicIMap imap(
    {grid_dim[0], grid_dim[1], grid_dim[2], forloop_range},
    get_data_dims(dtensor),
    next_dim_variable_index
  );
  {
    std::unordered_set<TensorDimConstraint> constraints;
    // Each tensor dim should be mapped to at most one device dim
    for (size_t i = 0; i < dtensor.dims.size(); ++i) {
      auto num_mapped = imap.mat.at({grid_dim[0], i}) + imap.mat.at({grid_dim[1], i}) + imap.mat.at({grid_dim[2], i});
      constraints.insert(num_mapped <= 1);
      constraints.insert(num_mapped >= (int)0);
    }
    // Each device dim should be mapped to at most one tensor dim
    for (SymbolicTensorDim const &device_dim : imap.parallel_dims) {
      std::shared_ptr<TensorDimExpr const> num_mapped = imap.mat.at({device_dim, 0});
      for (size_t i = 1; i < dtensor.dims.size(); ++i) {
        num_mapped = num_mapped + imap.mat.at({device_dim, i});
      }
      constraints.insert(num_mapped <= 1);
      constraints.insert(num_mapped >= (int)0);
    }
    if (!conds.add_constraints(constraints)) {
      return false;
    }
  }

  std::shared_ptr<OpArgs const> args =
      std::make_shared<TBInputOpArgs const>(dtensor, imap);
  SymbolicTBOp op(type::TBOperatorType::TB_INPUT_OP, args);

  auto compute_symbolic_stensor = [&](SymbolicDTensor const &dtensor,
                                     SymbolicIMap const &imap) {
    std::vector<SymbolicTensorDim> dim_templates = dtensor.dims;
    assert(dim_templates.size() == imap.data_dims.size());
    for (size_t i = 0; i < dim_templates.size(); ++i) {
      SymbolicTensorDim divisor = grid_dim[0] * imap.mat.at({grid_dim[0], i}) + grid_dim[1] * imap.mat.at({grid_dim[1], i}) + grid_dim[2] * imap.mat.at({grid_dim[2], i});
      divisor = divisor * (forloop_range * imap.mat.at({forloop_range, i}));
      dim_templates[i] = dim_templates[i] / divisor;
    }
    return SymbolicSTensor(dim_templates, false);
  };

  SymbolicSTensor tensor = compute_symbolic_stensor(dtensor, imap);

  operators.push_back(op);
  tensors.push_back(tensor);
  input_indices.push_back({});
  output_indices.push_back({(int)tensors.size() - 1});

  return true;
}

bool SymbolicTBGraph::add_output(int input_index, mirage::type::TBEpilogueType epilogue_type) {
  auto get_data_dims = [](SymbolicSTensor const &stensor) {
    std::vector<size_t> data_dims;
    for (size_t i = 0; i < stensor.dims.size(); ++i) {
      data_dims.push_back(i);
    }
    return data_dims;
  };
  SymbolicOmap omap(
    {grid_dim[0], grid_dim[1], grid_dim[2]},
    get_data_dims(this->tensors[input_index]),
    next_dim_variable_index
  );
  {
    std::unordered_set<TensorDimConstraint> constraints;
    // Each tensor dim should be mapped to at most one device dim
    for (size_t i = 0; i < tensors[input_index].dims.size(); ++i) {
      auto num_mapped = omap.mat.at({grid_dim[0], i}) + omap.mat.at({grid_dim[1], i}) + omap.mat.at({grid_dim[2], i});
      constraints.insert(num_mapped <= 1);
      constraints.insert(num_mapped >= (int)0);
    }
    // Each device dim should be mapped to at most one tensor dim
    for (SymbolicTensorDim const &device_dim : omap.parallel_dims) {
      std::shared_ptr<TensorDimExpr const> num_mapped = omap.mat.at({device_dim, 0});
      for (size_t i = 1; i < tensors[input_index].dims.size(); ++i) {
        num_mapped = num_mapped + omap.mat.at({device_dim, i});
      }
      constraints.insert(dim_expr_make_ite(device_dim == 1, num_mapped == 0, num_mapped == 1));
    }
    if (!conds.add_constraints(constraints)) {
      return false;
    }
  }

  auto compute_symbolic_dtensor = [&](SymbolicSTensor const &stensor,
                                     SymbolicOmap const &omap) {
    std::vector<SymbolicTensorDim> dim_templates = stensor.dims;
    assert(dim_templates.size() == omap.data_dims.size());
    for (size_t i = 0; i < dim_templates.size(); ++i) {
      SymbolicTensorDim multiplier = grid_dim[0] * omap.mat.at({grid_dim[0], i}) + grid_dim[1] * omap.mat.at({grid_dim[1], i}) + grid_dim[2] * omap.mat.at({grid_dim[2], i});
      dim_templates[i] = dim_templates[i] * multiplier;
    }
    return SymbolicDTensor(dim_templates);
  };

  SymbolicDTensor dtensor = compute_symbolic_dtensor(this->tensors[input_index], omap);
  std::shared_ptr<OpArgs const> args =
      std::make_shared<TBOutputOpArgs const>(dtensor, omap, epilogue_type);
  SymbolicTBOp op(type::TBOperatorType::TB_OUTPUT_OP, args);
  operators.push_back(op);
  input_indices.push_back({input_index});
  output_indices.push_back({});
  return true;
}

mirage::kernel::Graph *SymbolicKNGraph::to_kernel_graph(
    DimVarAssignments const &assignments) const {
  kernel::Graph *graph = new kernel::Graph();
  std::vector<kernel::DTensor> tensors_val;
  for (size_t i = 0; i < this->operators.size(); ++i) {
    std::vector<kernel::DTensor> input_tensors;
    for (int input_index : this->input_indices[i]) {
      input_tensors.push_back(tensors_val[input_index]);
    }
    kernel::KNOperator *op = nullptr;
    if (this->operators[i].op_type == type::KNOperatorType::KN_CUSTOMIZED_OP) {
      std::shared_ptr<KNCustomizedOpArgs const> args =
          std::static_pointer_cast<KNCustomizedOpArgs const>(
              this->operators[i].args);
      threadblock::Graph *tb_graph =
          args->tb_graph_template.to_threadblock_graph(assignments,
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
    for (DTensor const &output_tensor : op->output_tensors) {
      tensors_val.push_back(output_tensor);
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
  conds.revert();
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
        std::unordered_set<TensorDimConstraint> constraints;
        for (size_t i = 0; i < A.dims.size(); i++) {
          constraints.insert(
              dim_expr_make_disj(
                {A.dims[i] == B.dims[i], A.dims[i] == 1, B.dims[i] == 1}
              ));
        }
        if (!conds.add_constraints(constraints)) {
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
      if (!conds.add_constraints({})) {
        return false;
      }
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
        std::unordered_set<TensorDimConstraint> constraints;
        for (size_t i = 0; i < A.dims.size() - 2; i++) {
          constraints.insert(A.dims[i] == B.dims[i]);
        }
        constraints.insert(A.dims[A.dims.size() - 1] == B.dims[B.dims.size() - 2]);
        if (!conds.add_constraints(constraints)) {
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
      if (!conds.add_constraints({})) {
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
  assert(tb_graph.conds.add_constraint(tb_graph.get_memory_usage_constraint()));
  // std::cerr << "Checking tb_graph conds" << std::endl;
  // std::cerr << json(tb_graph) << std::endl;
  // if (!tb_graph.conds.satisfiable()) {
  //   std::cerr << json(tb_graph) << std::endl;
  //   std::cerr << "tb_graph conds not satisfiable" << std::endl;
  //   return false;
  // }
  this->operators.push_back(
      SymbolicKNOp(type::KNOperatorType::KN_CUSTOMIZED_OP,
                   std::make_shared<KNCustomizedOpArgs>(tb_graph)));
  if (!conds.add_constraints(tb_graph.conds.get_all_constraints())) {
    return false;
  }
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
                                int3 input_map) {
  this->operators.push_back(
      SymbolicKNOp(type::KNOperatorType::KN_INPUT_OP,
                   std::make_shared<KNInputOpArgs>(input_strides, input_map)));
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
  if (!conds.add_constraints({})) {
    return false;
  }
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
  return json{
      {"grid_dim", grid_dim_json},
      {"block_dim", block_dim_json},
      {"forloop_range", *forloop_range},
      {"reduction_dimx", reduction_dimx},
      {"operators", operators},
      {"tensors", tensors},
      {"input_indices", input_indices},
      {"output_indices", output_indices},
      {"conds", conds},
  };
}

SymbolicKNGraph::operator json() const {
  return json{
      {"operators", operators},
      {"tensors", tensors},
      {"input_indices", input_indices},
      {"output_indices", output_indices},
      {"conds", conds},
  };
}

} // namespace search
} // namespace mirage
