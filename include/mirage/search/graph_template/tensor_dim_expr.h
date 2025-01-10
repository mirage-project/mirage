#pragma once

#include <memory>
#include <string>

namespace mirage {
namespace search {

class TensorDimExpr {
public:
  TensorDimExpr() = default;
  virtual ~TensorDimExpr() = default;
};

using tensor_dim_var_index_t = uint32_t;

class TensorDimVar : public TensorDimExpr {
public:
  TensorDimVar(tensor_dim_var_index_t index);
  tensor_dim_var_index_t index;
};

class TensorDimConst : public TensorDimExpr {
public:
  TensorDimConst(int value);
  int value;
};

class TensorDimAdd : public TensorDimExpr {
public:
  TensorDimAdd(std::shared_ptr<TensorDimExpr> lhs, std::shared_ptr<TensorDimExpr> rhs);
  std::shared_ptr<TensorDimExpr> lhs, rhs;
};

class TensorDimMul : public TensorDimExpr {
public:
  TensorDimMul(std::shared_ptr<TensorDimExpr> lhs, std::shared_ptr<TensorDimExpr> rhs);
  std::shared_ptr<TensorDimExpr> lhs, rhs;
};

class TensorDimDiv : public TensorDimExpr {
public:
  TensorDimDiv(std::shared_ptr<TensorDimExpr> lhs, std::shared_ptr<TensorDimExpr> rhs);
  std::shared_ptr<TensorDimExpr> lhs, rhs;
};

}
}
