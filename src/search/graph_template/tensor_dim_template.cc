#include "mirage/search/graph_template/symbolic_tensor_dim.h"

namespace mirage {
namespace search {

SymbolicTensorDim::SymbolicTensorDim(std::shared_ptr<TensorDimExpr> dim_expr) : dim_expr(dim_expr) {}

}
}
