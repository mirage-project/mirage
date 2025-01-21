#pragma once

#include "mirage/search/symbolic_graph/tensor_dim_expr.h"

namespace mirage {
namespace search {

class SymbolicTensorDim {
public:
  SymbolicTensorDim(std::shared_ptr<TensorDimExpr> dim_expr);

  std::shared_ptr<TensorDimExpr> dim_expr;

  operator json() const;
};

}
}
