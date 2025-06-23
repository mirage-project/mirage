#include "mirage/search/symbolic_graph/tensor_dim_constraint.h"
#include "mirage/utils/hash_utils.h"

#include <unordered_map>

namespace mirage {
namespace search {

TensorDimConstraint::TensorDimConstraint(ConstraintType type,
                                         std::vector<SymbolicTensorDim> dims)
    : type(type), dims(dims) {}

TensorDimConstraint::operator json() const {
  return json{
      {"type", type},
      {"dims", dims},
  };
}

bool TensorDimConstraint::operator==(TensorDimConstraint const &other) const {
  return type == other.type && dims == other.dims;
}

z3::expr TensorDimConstraint::to_z3(z3::context &c,
                                    DimVarAssignments const &assign) const {
  std::vector<z3::expr> exprs;
  for (SymbolicTensorDim const &dim : dims) {
    exprs.push_back(dim.dim_expr->to_z3(c, assign));
  }

  switch (type) {
    case ConstraintType::EQUAL: {
      return exprs[0] == exprs[1];
    }
    case ConstraintType::EQUAL_OR_ONE: {
      return exprs[0] == exprs[1] || exprs[0] == 1 || exprs[1] == 1;
    }
    case ConstraintType::GEQ: {
      return exprs[0] >= exprs[1];
    }
    case ConstraintType::LEQ: {
      return exprs[0] <= exprs[1];
    }
    default:
      assert(false && "Unsupported constraint type");
  }
}

TensorDimConstraint make_equal_constraint(SymbolicTensorDim lhs,
                                          SymbolicTensorDim rhs) {
  return TensorDimConstraint(ConstraintType::EQUAL, {lhs, rhs});
}

TensorDimConstraint make_equal_or_one_constraint(SymbolicTensorDim lhs,
                                                 SymbolicTensorDim rhs) {
  return TensorDimConstraint(ConstraintType::EQUAL_OR_ONE, {lhs, rhs});
}

TensorDimConstraint make_non_negative_constraint(SymbolicTensorDim dim) {
  SymbolicTensorDim zero(dim_expr_make_const(0));
  return TensorDimConstraint(ConstraintType::GEQ, {dim, zero});
}

TensorDimConstraint
    make_sum_leq_one_constraint(std::vector<SymbolicTensorDim> dims) {
  dims.push_back(SymbolicTensorDim(std::make_shared<TensorDimConst const>(-1)));
  std::shared_ptr<TensorDimExpr const> sum_expr = dims[0].dim_expr;
  for (size_t i = 1; i < dims.size(); ++i) {
    sum_expr = std::make_shared<TensorDimAdd>(sum_expr, dims[i].dim_expr);
  }
  SymbolicTensorDim sum(sum_expr);
  SymbolicTensorDim zero(dim_expr_make_const(0));
  return TensorDimConstraint(ConstraintType::LEQ, {sum, zero});
}

TensorDimConstraint
    make_sum_geq_zero_constraint(std::vector<SymbolicTensorDim> dims) {
  assert(!dims.empty());
  std::shared_ptr<TensorDimExpr const> sum_expr = dims[0].dim_expr;
  for (size_t i = 1; i < dims.size(); ++i) {
    sum_expr = std::make_shared<TensorDimAdd>(sum_expr, dims[i].dim_expr);
  }
  SymbolicTensorDim sum(sum_expr);
  SymbolicTensorDim zero(dim_expr_make_const(0));
  return TensorDimConstraint(ConstraintType::GEQ, {sum, zero});
}

bool check_satisfiability(
    std::unordered_set<TensorDimConstraint> const &pre_conds,
    std::unordered_set<TensorDimConstraint> const &constraints) {
  auto probably_equal = [](std::shared_ptr<TensorDimExpr const> el,
                           std::shared_ptr<TensorDimExpr const> er) {
    {
      std::shared_ptr<TensorDimConst const>
          cl = std::dynamic_pointer_cast<TensorDimConst const>(el),
          cr = std::dynamic_pointer_cast<TensorDimConst const>(er);
      if (cl && cr) {
        if (cl->value != cr->value) {
          return false;
        }
      }
    }
    {
      std::shared_ptr<TensorDimDiv const>
          dl = std::dynamic_pointer_cast<TensorDimDiv const>(el),
          dr = std::dynamic_pointer_cast<TensorDimDiv const>(er);
      if (dl && dr) {
        std::shared_ptr<TensorDimConst const>
            cll = std::dynamic_pointer_cast<TensorDimConst const>(dl->lhs),
            crl = std::dynamic_pointer_cast<TensorDimConst const>(dr->lhs);
        std::shared_ptr<TensorDimVar const>
            clr = std::dynamic_pointer_cast<TensorDimVar const>(dl->rhs),
            crr = std::dynamic_pointer_cast<TensorDimVar const>(dr->rhs);
        if (cll && crl && clr && crr) {
          if (cll->value != crl->value || clr->index == crr->index) {
            return false;
          }
        }
      }
    }
    return true;
  };

  for (TensorDimConstraint const &constraint : constraints) {
    // rule-based checking for now
    if (constraint.type == ConstraintType::EQUAL) {
      return probably_equal(constraint.dims[0].dim_expr,
                            constraint.dims[1].dim_expr);
    }
    if (constraint.type == ConstraintType::EQUAL_OR_ONE) {
      std::shared_ptr<TensorDimExpr const> el = constraint.dims[0].dim_expr,
                                           er = constraint.dims[1].dim_expr;
      if (probably_equal(el, er)) {
        return true;
      }
      {
        std::shared_ptr<TensorDimConst const>
            cl = std::dynamic_pointer_cast<TensorDimConst const>(el),
            cr = std::dynamic_pointer_cast<TensorDimConst const>(er);
        if (cl && cr) {
          if (cl->value == 1 || cr->value == 1) {
            return true;
          }
        }
      }
    }
  }
  return true;
}

} // namespace search
} // namespace mirage

namespace std {

size_t hash<mirage::search::TensorDimConstraint>::operator()(
    mirage::search::TensorDimConstraint const &constraint) const {
  size_t seed = 0;
  hash_combine(seed, constraint.type);
  hash_combine(seed, constraint.dims.size());
  for (mirage::search::SymbolicTensorDim const &dim : constraint.dims) {
    hash_combine(seed, dim);
  }
  return seed;
}

} // namespace std