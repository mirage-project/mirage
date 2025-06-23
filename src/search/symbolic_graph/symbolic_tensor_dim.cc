#include "mirage/search/symbolic_graph/symbolic_tensor_dim.h"

namespace mirage {
namespace search {

SymbolicTensorDim::SymbolicTensorDim(
    std::shared_ptr<TensorDimExpr const> dim_expr)
    : dim_expr(dim_expr) {}

SymbolicTensorDim::operator json() const {
  return json{{"dim_expr", *dim_expr}};
}

bool SymbolicTensorDim::operator==(SymbolicTensorDim const &other) const {
  return dim_expr->same_expr_as(other.dim_expr);
}

SymbolicTensorDim
    SymbolicTensorDim::operator+(SymbolicTensorDim const &other) const {
  return SymbolicTensorDim(dim_expr_make_add(dim_expr, other.dim_expr));
}

SymbolicTensorDim
    SymbolicTensorDim::operator*(SymbolicTensorDim const &other) const {
  return SymbolicTensorDim(dim_expr_make_mul(dim_expr, other.dim_expr));
}

SymbolicTensorDim
    SymbolicTensorDim::operator/(SymbolicTensorDim const &other) const {
  return SymbolicTensorDim(dim_expr_make_div(dim_expr, other.dim_expr));
}

SymbolicTensorDim
    SymbolicTensorDim::operator^(SymbolicTensorDim const &other) const {
  return SymbolicTensorDim(dim_expr_make_pow(dim_expr, other.dim_expr));
}

} // namespace search
} // namespace mirage

namespace std {

size_t hash<mirage::search::SymbolicTensorDim>::operator()(
    mirage::search::SymbolicTensorDim const &dim) const {
  return dim.dim_expr->hash();
}

} // namespace std
