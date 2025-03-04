#pragma once

#include "mirage/search/symbolic_graph/tensor_dim_expr.h"

namespace mirage {
namespace search {

class SymbolicTensorDim {
public:
  SymbolicTensorDim(std::shared_ptr<TensorDimExpr const> dim_expr);

  std::shared_ptr<TensorDimExpr const> dim_expr;

  operator json() const;
  bool operator==(SymbolicTensorDim const &other) const;

  SymbolicTensorDim operator+(SymbolicTensorDim const &other) const;
  SymbolicTensorDim operator*(SymbolicTensorDim const &other) const;
  SymbolicTensorDim operator/(SymbolicTensorDim const &other) const;
  SymbolicTensorDim operator^(SymbolicTensorDim const &other) const;
};

} // namespace search
} // namespace mirage

namespace std {

template <>
struct hash<mirage::search::SymbolicTensorDim> {
  size_t operator()(mirage::search::SymbolicTensorDim const &dim) const;
};

} // namespace std
