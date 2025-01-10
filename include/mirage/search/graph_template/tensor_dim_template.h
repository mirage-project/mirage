#pragma once

#include "mirage/search/graph_template/tensor_dim_expr.h"

namespace mirage {
namespace search {

class TensorDimTemplate {
public:
  TensorDimTemplate(std::shared_ptr<TensorDimExpr> dim_expr);

  std::shared_ptr<TensorDimExpr> dim_expr;
};

}
}
