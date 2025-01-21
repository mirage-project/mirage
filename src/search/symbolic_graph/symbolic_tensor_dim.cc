#include "mirage/search/symbolic_graph/symbolic_tensor_dim.h"

namespace mirage {
namespace search {

SymbolicTensorDim::SymbolicTensorDim(std::shared_ptr<TensorDimExpr> dim_expr) : dim_expr(dim_expr) {}

SymbolicTensorDim::operator json() const {
  return json{{"dim_expr", *dim_expr}};
}

}
}
