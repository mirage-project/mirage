#pragma once

#include "mirage/search/graph_template/dim_var_assignments.h"

#include <memory>
#include <string>

namespace mirage {
namespace search {

class TensorDimExpr {
public:
  TensorDimExpr() = default;
  virtual ~TensorDimExpr() = default;

  virtual int get_value(DimVarAssignments const &assignments) const = 0;
};

class TensorDimVar : public TensorDimExpr {
public:
  TensorDimVar(tensor_dim_var_index_t index);
  tensor_dim_var_index_t index;

  int get_value(DimVarAssignments const &assignments) const override;
};

class TensorDimConst : public TensorDimExpr {
public:
  TensorDimConst(int value);
  int value;

  int get_value(DimVarAssignments const &assignments) const override;
};

class TensorDimAdd : public TensorDimExpr {
public:
  TensorDimAdd(std::shared_ptr<TensorDimExpr> lhs, std::shared_ptr<TensorDimExpr> rhs);
  std::shared_ptr<TensorDimExpr> lhs, rhs;

  int get_value(DimVarAssignments const &assignments) const override;
};

class TensorDimMul : public TensorDimExpr {
public:
  TensorDimMul(std::shared_ptr<TensorDimExpr> lhs, std::shared_ptr<TensorDimExpr> rhs);
  std::shared_ptr<TensorDimExpr> lhs, rhs;

  int get_value(DimVarAssignments const &assignments) const override;
};

class TensorDimDiv : public TensorDimExpr {
public:
  TensorDimDiv(std::shared_ptr<TensorDimExpr> lhs, std::shared_ptr<TensorDimExpr> rhs);
  std::shared_ptr<TensorDimExpr> lhs, rhs;

  int get_value(DimVarAssignments const &assignments) const override;
};

}
}
