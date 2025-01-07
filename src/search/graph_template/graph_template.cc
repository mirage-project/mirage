#include "mirage/search/graph_template/graph_template.h"

namespace mirage {
namespace search {

TBGraphTemplate::TBGraphTemplate() : grid_dim(1, 1, 1), block_dim(1, 1, 1), forloop_range(1), reduction_dimx(1) {}

mirage::threadblock::Graph TBGraphTemplate::to_threadblock_graph() {
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
      if (A.num_dims != B.num_dims) {
        return false;
      }
      if (concat_dim > A.num_dims) {
        return false;
      }
      {
        std::vector<TensorDimConstraint> constraints;
        for (int i = 0; i < A.num_dims; i++) {
          if (i != concat_dim) {
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
        for (int i = 0; i < A.num_dims; i++) {
          if (i != concat_dim) {
            dim_templates.push_back(A.dims[i]);
          } else {
            std::shared_ptr<TensorDimExpr> dim_expr = std::make_shared<TensorDimAdd>(A.dims[i].expr, B.dims[i].expr);
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
      if (A.num_dims != B.num_dims) {
        return false;
      }
      {
        std::vector<TensorDimConstraint> constraints;
        for (int i = 0; i < A.num_dims; i++) {
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
        for (int i = 0; i < A.num_dims; i++) {
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
      if (A.num_dims != B.num_dims) {
        return false;
      }
      {
        std::vector<TensorDimConstraint> constraints;
        for (int i = 0; i < A.num_dims - 2; i++) {
          constraints.push_back(make_equal_constraint(A.dims[i], TensorDimTemplate(std::make_shared<TensorDimConst>(1))));
          constraints.push_back(make_equal_constraint(B.dims[i], TensorDimTemplate(std::make_shared<TensorDimConst>(1))));
        }
        constraints.push_back(make_equal_constraint(A.dims[A.num_dims - 1], B.dims[B.num_dims - 2]));
        if (!check_satisfiability(this->conds, constraints)) {
          return false;
        }
        conds.insert(conds.end(), constraints.begin(), constraints.end());
      }
      this->operators.push_back(TBOpTemplate(op_type));
      {
        std::vector<TensorDimTemplate> dim_templates;
        for (int i = 0; i < A.num_dims; i++) {
          dim_templates.push_back(A.dims[i]);
        }
        dim_templates[dim_templates.size() - 1] = B.dims[B.num_dims - 1];
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
      if (A.num_dims <= reduction_dim) {
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
      if (A.num_dims <= reduction_dim) {
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
    case type::TBOperatorType::TB_OUTPUT_OP: {
    }
    default: {
      return false;
    }
  }
}

}
}
