#include "mirage/search/graph_template/graph_template.h"
#include "mirage/search/graph_template/op_args.h"

namespace mirage {
namespace search {

tensor_dim_var_index_t TBGraphTemplate::next_dim_variable_index = 0;

TBGraphTemplate::TBGraphTemplate() :
  grid_dim({TensorDimTemplate(std::make_shared<TensorDimVar>(next_dim_variable_index++)),
              TensorDimTemplate(std::make_shared<TensorDimVar>(next_dim_variable_index++)),
              TensorDimTemplate(std::make_shared<TensorDimVar>(next_dim_variable_index++))}),
  block_dim({TensorDimTemplate(std::make_shared<TensorDimConst>(128)),
               TensorDimTemplate(std::make_shared<TensorDimConst>(1)),
               TensorDimTemplate(std::make_shared<TensorDimConst>(1))}),
  forloop_range(TensorDimTemplate(std::make_shared<TensorDimVar>(next_dim_variable_index++)))
{}

mirage::threadblock::Graph TBGraphTemplate::to_threadblock_graph(std::unordered_map<tensor_dim_var_index_t, int> const &assignment) {
  // TODO
}

bool TBGraphTemplate::add_operator(type::TBOperatorType op_type, std::vector<int> input_indices) {
  for (size_t i = 1; i < input_indices.size(); i++) {
    if (tensors[input_indices[i]].after_accum != tensors[input_indices[0]].after_accum) {
      return false;
    }
  }

  switch (op_type) {
    case type::TBOperatorType::TB_CONCAT_0_OP:
    case type::TBOperatorType::TB_CONCAT_1_OP:
    case type::TBOperatorType::TB_CONCAT_2_OP: {
      int concat_dim = (int)op_type - (int)type::TBOperatorType::TB_CONCAT_0_OP;
      assert(input_indices.size() == 2);
      STensorTemplate A = this->tensors[input_indices[0]];
      STensorTemplate B = this->tensors[input_indices[1]];
      if (A.dims.size() != B.dims.size()) {
        return false;
      }
      if (concat_dim > (int)A.dims.size()) {
        return false;
      }
      {
        std::vector<TensorDimConstraint> constraints;
        for (size_t i = 0; i < A.dims.size(); i++) {
          if ((int)i != concat_dim) {
            constraints.push_back(make_equal_constraint(A.dims[i], B.dims[i]));
          }
        }
        if (!check_satisfiability(this->conds, constraints)) {
          return false;
        }
        conds.insert(conds.end(), constraints.begin(), constraints.end());
      }
      this->operators.push_back(TBOpTemplate(op_type, std::make_shared<TBConcatOpArgs>(concat_dim)));
      {
        std::vector<TensorDimTemplate> dim_templates;
        for (size_t i = 0; i < A.dims.size(); i++) {
          if ((int)i != concat_dim) {
            dim_templates.push_back(A.dims[i]);
          } else {
            std::shared_ptr<TensorDimExpr> dim_expr = std::make_shared<TensorDimAdd>(A.dims[i].dim_expr, B.dims[i].dim_expr);
            dim_templates.push_back(TensorDimTemplate(dim_expr));
          }
        }
        STensorTemplate C(dim_templates, A.after_accum);
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
      STensorTemplate A = this->tensors[input_indices[0]];
      this->operators.push_back(TBOpTemplate(op_type));
      this->tensors.push_back(A);
      this->input_indices.push_back(input_indices);
      this->output_indices.push_back({(int)this->tensors.size() - 1});
      break;
    }
    case type::TBOperatorType::TB_ADD_OP:
    case type::TBOperatorType::TB_MUL_OP:
    case type::TBOperatorType::TB_DIV_OP: {
      assert(input_indices.size() == 2);
      STensorTemplate A = this->tensors[input_indices[0]];
      STensorTemplate B = this->tensors[input_indices[1]];
      if (A.dims.size() != B.dims.size()) {
        return false;
      }
      {
        std::vector<TensorDimConstraint> constraints;
        for (size_t i = 0; i < A.dims.size(); i++) {
          constraints.push_back(make_equal_or_one_constraint(A.dims[i], B.dims[i]));
        }
        if (!check_satisfiability(this->conds, constraints)) {
          return false;
        }
        conds.insert(conds.end(), constraints.begin(), constraints.end());
      }
      this->operators.push_back(TBOpTemplate(op_type));
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
      STensorTemplate A = this->tensors[input_indices[0]];
      if (A.after_accum) {
        return false;
      }
      this->operators.push_back(TBOpTemplate(op_type));
      {
        std::vector<TensorDimTemplate> dim_templates;
        for (size_t i = 0; i < A.dims.size(); i++) {
          dim_templates.push_back(A.dims[i]);
        }
        if (op_type == type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_SUM_OP ||
            op_type == type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_MEAN_OP ||
            op_type == type::TBOperatorType::TB_FORLOOP_ACCUM_RED_LD_RMS_OP) {
          dim_templates[dim_templates.size() - 1] = TensorDimTemplate(std::make_shared<TensorDimConst>(1));
        } else if (op_type == type::TBOperatorType::TB_FORLOOP_ACCUM_REDTOX_LD_SUM_OP) {
          dim_templates[dim_templates.size() - 1] = TensorDimTemplate(std::make_shared<TensorDimConst>(this->reduction_dimx));
        }
        STensorTemplate B(dim_templates, true);
        this->tensors.push_back(B);
      }
      this->input_indices.push_back(input_indices);
      this->output_indices.push_back({(int)this->tensors.size() - 1});
      break;
    }
    case type::TBOperatorType::TB_MATMUL_OP: {
      assert(input_indices.size() == 2);
      STensorTemplate A = this->tensors[input_indices[0]];
      STensorTemplate B = this->tensors[input_indices[1]];
      if (A.dims.size() != B.dims.size()) {
        return false;
      }
      {
        std::vector<TensorDimConstraint> constraints;
        for (size_t i = 0; i < A.dims.size() - 2; i++) {
          constraints.push_back(make_equal_constraint(A.dims[i], TensorDimTemplate(std::make_shared<TensorDimConst>(1))));
          constraints.push_back(make_equal_constraint(B.dims[i], TensorDimTemplate(std::make_shared<TensorDimConst>(1))));
        }
        constraints.push_back(make_equal_constraint(A.dims[A.dims.size() - 1], B.dims[B.dims.size() - 2]));
        if (!check_satisfiability(this->conds, constraints)) {
          return false;
        }
        conds.insert(conds.end(), constraints.begin(), constraints.end());
      }
      this->operators.push_back(TBOpTemplate(op_type));
      {
        std::vector<TensorDimTemplate> dim_templates;
        for (size_t i = 0; i < A.dims.size(); i++) {
          dim_templates.push_back(A.dims[i]);
        }
        dim_templates[dim_templates.size() - 1] = B.dims[B.dims.size() - 1];
        STensorTemplate C(dim_templates, A.after_accum);
        this->tensors.push_back(C);
      }
      this->input_indices.push_back(input_indices);
      this->output_indices.push_back({(int)this->tensors.size() - 1});
      break;
    }
    case type::TBOperatorType::TB_REDUCTION_0_OP:
    case type::TBOperatorType::TB_REDUCTION_1_OP:
    case type::TBOperatorType::TB_REDUCTION_2_OP: {
      int reduction_dim = (int)op_type - (int)type::TBOperatorType::TB_REDUCTION_0_OP;
      assert(input_indices.size() == 1);
      STensorTemplate A = this->tensors[input_indices[0]];
      if ((int)A.dims.size() <= reduction_dim) {
        return false;
      }
      this->operators.push_back(TBOpTemplate(op_type, std::make_shared<TBReductionOpArgs>(reduction_dim, 1)));
      {
        std::vector<TensorDimTemplate> dim_templates = A.dims;
        dim_templates[reduction_dim] = TensorDimTemplate(std::make_shared<TensorDimConst>(1));
        STensorTemplate B(dim_templates, A.after_accum);
        this->tensors.push_back(B);
      }
      this->input_indices.push_back(input_indices);
      this->output_indices.push_back({(int)this->tensors.size() - 1});
      break;
    }
    case type::TBOperatorType::TB_REDUCTION_0_TO_DIMX_OP:
    case type::TBOperatorType::TB_REDUCTION_1_TO_DIMX_OP:
    case type::TBOperatorType::TB_REDUCTION_2_TO_DIMX_OP: {
      int reduction_dim = (int)op_type - (int)type::TBOperatorType::TB_REDUCTION_0_TO_DIMX_OP;
      assert(input_indices.size() == 1);
      STensorTemplate A = this->tensors[input_indices[0]];
      if ((int)A.dims.size() <= reduction_dim) {
        return false;
      }
      this->operators.push_back(TBOpTemplate(op_type, std::make_shared<TBReductionOpArgs>(reduction_dim, this->reduction_dimx)));
      {
        std::vector<TensorDimTemplate> dim_templates = A.dims;
        dim_templates[reduction_dim] = TensorDimTemplate(std::make_shared<TensorDimConst>(this->reduction_dimx));
        STensorTemplate B(dim_templates, A.after_accum);
        this->tensors.push_back(B);
      }
      this->input_indices.push_back(input_indices);
      this->output_indices.push_back({(int)this->tensors.size() - 1});
      break;
    }
    case type::TBOperatorType::TB_RMS_NORM_OP: {
      assert(input_indices.size() == 1);
      STensorTemplate A = this->tensors[input_indices[0]];
      this->operators.push_back(TBOpTemplate(op_type));
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

bool TBGraphTemplate::add_input(DTensorTemplate dtensor, int3 input_map, int forloop_dim) {
  // create op template
  std::shared_ptr<OpArgs> args = std::make_shared<TBInputOpArgs>(dtensor, input_map, forloop_dim);
  TBOpTemplate op(type::TBOperatorType::TB_INPUT_OP, args);
  operators.push_back(op);
  // create stensor template
  std::vector<TensorDimTemplate> dim_templates = dtensor.dims;
  for (int d = 0; d < 3; ++d) {
    int dim_idx = -1;
    if (d == 0) {
      dim_idx = input_map.x;
    }
    if (d == 1) {
      dim_idx = input_map.y;
    }
    if (d == 2) {
      dim_idx = input_map.z;
    }
    if (dim_idx >= 0) {
      dim_templates.push_back(TensorDimTemplate(std::make_shared<TensorDimDiv>(dim_templates[dim_idx].dim_expr, grid_dim[d].dim_expr)));
    }
  }
  if (forloop_dim >= 0) {
    dim_templates[forloop_dim] = TensorDimTemplate(std::make_shared<TensorDimDiv>(dim_templates[forloop_dim].dim_expr, forloop_range.dim_expr));
  }
  STensorTemplate tensor(dim_templates, false);
  tensors.push_back(tensor);
  input_indices.push_back({});
  output_indices.push_back({(int)tensors.size() - 1});
  return true;
}

bool TBGraphTemplate::add_output(int input_index, int3 output_map, int forloop_dim, mirage::type::TBEpilogueType epilogue_type) {
  if (!tensors[input_index].after_accum) {
    return false;
  }
  // create dtensor template
  std::vector<TensorDimTemplate> dim_templates = tensors[input_index].dims;
  for (int d = 0; d < 3; ++d) {
    int dim_idx = -1;
    if (d == 0) {
      dim_idx = output_map.x;
    }
    if (d == 1) {
      dim_idx = output_map.y;
    }
    if (d == 2) {
      dim_idx = output_map.z;
    }
    if (dim_idx >= 0) {
      dim_templates[dim_idx] = TensorDimTemplate(std::make_shared<TensorDimMul>(dim_templates[dim_idx].dim_expr, grid_dim[d].dim_expr));
    }
  }
  if (forloop_dim >= 0) {
    dim_templates[forloop_dim] = TensorDimTemplate(std::make_shared<TensorDimMul>(dim_templates[forloop_dim].dim_expr, forloop_range.dim_expr));
  }
  DTensorTemplate dtensor(dim_templates);

  // create op template
  std::shared_ptr<OpArgs> args = std::make_shared<TBOutputOpArgs>(dtensor, output_map, forloop_dim, epilogue_type);
  TBOpTemplate op(type::TBOperatorType::TB_OUTPUT_OP, args);
  operators.push_back(op);
  input_indices.push_back({input_index});
  output_indices.push_back({});
  return true;
}

mirage::kernel::Graph KNGraphTemplate::to_kernel_graph(std::unordered_map<tensor_dim_var_index_t, int> const &assignment) {
  // TODO
}

bool KNGraphTemplate::add_operator(type::KNOperatorType op_type, std::vector<int> input_indices) {
  switch (op_type) {
    case type::KNOperatorType::KN_ADD_OP:
    case type::KNOperatorType::KN_MUL_OP:
    case type::KNOperatorType::KN_DIV_OP: {
      assert(input_indices.size() == 2);
      DTensorTemplate A = this->tensors[input_indices[0]];
      DTensorTemplate B = this->tensors[input_indices[1]];
      if (A.dims.size() != B.dims.size()) {
        return false;
      }
      {
        std::vector<TensorDimConstraint> constraints;
        for (size_t i = 0; i < A.dims.size(); i++) {
          constraints.push_back(make_equal_or_one_constraint(A.dims[i], B.dims[i]));
        }
        if (!check_satisfiability(this->conds, constraints)) {
          return false;
        }
        conds.insert(conds.end(), constraints.begin(), constraints.end());
      }
      this->operators.push_back(KNOpTemplate(op_type));
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
      DTensorTemplate A = this->tensors[input_indices[0]];
      this->operators.push_back(KNOpTemplate(op_type));
      this->tensors.push_back(A);
      this->input_indices.push_back(input_indices);
      this->output_indices.push_back({(int)this->tensors.size() - 1});
      break;
    }
    case type::KNOperatorType::KN_MATMUL_OP: {
      assert(input_indices.size() == 2);
      DTensorTemplate A = this->tensors[input_indices[0]];
      DTensorTemplate B = this->tensors[input_indices[1]];
      if (A.dims.size() != B.dims.size()) {
        return false;
      }
      {
        std::vector<TensorDimConstraint> constraints;
        for (size_t i = 0; i < A.dims.size() - 2; i++) {
          constraints.push_back(make_equal_constraint(A.dims[i], B.dims[i]));
        }
        constraints.push_back(make_equal_constraint(A.dims[A.dims.size() - 1], B.dims[B.dims.size() - 2]));
        if (!check_satisfiability(this->conds, constraints)) {
          return false;
        }
        conds.insert(conds.end(), constraints.begin(), constraints.end());
      }
      this->operators.push_back(KNOpTemplate(op_type));
      {
        std::vector<TensorDimTemplate> dim_templates;
        for (size_t i = 0; i < A.dims.size(); i++) {
          dim_templates.push_back(A.dims[i]);
        }
        dim_templates[dim_templates.size() - 1] = B.dims[B.dims.size() - 1];
        DTensorTemplate C(dim_templates);
        this->tensors.push_back(C);
      }
      this->input_indices.push_back(input_indices);
      this->output_indices.push_back({(int)this->tensors.size() - 1});
      break;
    }
    case type::KNOperatorType::KN_REDUCTION_0_OP:
    case type::KNOperatorType::KN_REDUCTION_1_OP:
    case type::KNOperatorType::KN_REDUCTION_2_OP: {
      int reduction_dim = (int)op_type - (int)type::KNOperatorType::KN_REDUCTION_0_OP;
      int reduction_dim_size = 1;
      assert(input_indices.size() == 1);
      DTensorTemplate A = this->tensors[input_indices[0]];
      if ((int)A.dims.size() <= reduction_dim) {
        return false;
      }
      this->operators.push_back(KNOpTemplate(op_type, std::make_shared<KNReductionOpArgs>(reduction_dim, reduction_dim_size)));
      {
        std::vector<TensorDimTemplate> dim_templates = A.dims;
        dim_templates[reduction_dim] = TensorDimTemplate(std::make_shared<TensorDimConst>(reduction_dim_size));
        DTensorTemplate B(dim_templates);
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

bool KNGraphTemplate::add_customized_operator(TBGraphTemplate tb_graph, std::vector<int> input_indices) {
  this->operators.push_back(KNOpTemplate(type::KNOperatorType::KN_CUSTOMIZED_OP, std::make_shared<KNCustomizedOpArgs>(tb_graph)));
  std::vector<int> output_indices;
  for (auto const &op : tb_graph.operators) {
    if (op.op_type == type::TBOperatorType::TB_OUTPUT_OP) {
      std::shared_ptr<TBOutputOpArgs> args = std::dynamic_pointer_cast<TBOutputOpArgs>(op.args);
      output_indices.push_back((int)this->tensors.size());
      this->tensors.push_back(args->dtensor);
    }
  }
  this->input_indices.push_back(input_indices);
  this->output_indices.push_back(output_indices);
  return true;
}

bool KNGraphTemplate::add_input(std::vector<int> input_dims, std::vector<size_t> input_strides, int3 input_map) {
  this->operators.push_back(KNOpTemplate(type::KNOperatorType::KN_INPUT_OP, std::make_shared<KNInputOpArgs>(input_strides, input_map)));
  {
    std::vector<TensorDimTemplate> dim_templates;
    for (size_t i = 0; i < input_dims.size(); i++) {
      dim_templates.push_back(TensorDimTemplate(std::make_shared<TensorDimConst>(input_dims[i])));
    }
    DTensorTemplate tensor(dim_templates);
    this->tensors.push_back(tensor);
  }
  this->input_indices.push_back({});
  this->output_indices.push_back({(int)this->tensors.size() - 1});
  return true;
}

bool KNGraphTemplate::add_output(int input_index, std::vector<size_t> output_strides, int3 output_map) {
  this->operators.push_back(KNOpTemplate(type::KNOperatorType::KN_OUTPUT_OP, std::make_shared<KNOutputOpArgs>(output_strides, output_map)));
  this->input_indices.push_back({input_index});
  this->output_indices.push_back({});
  return true;
}

}
}
