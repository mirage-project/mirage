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
      if (input_indices[0] == input_indices[1]) {
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

} // namespace search
} // namespace mirage
