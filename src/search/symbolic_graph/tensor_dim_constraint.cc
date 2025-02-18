#include "mirage/search/symbolic_graph/tensor_dim_constraint.h"
#include "mirage/utils/hash_utils.h"

#include <unordered_map>

namespace mirage {
namespace search {

TensorDimConstraint::TensorDimConstraint(ConstraintType type,
                                         SymbolicTensorDim lhs,
                                         SymbolicTensorDim rhs)
    : type(type), lhs(lhs), rhs(rhs) {}

TensorDimConstraint::operator json() const {
  return json{
      {"type", type},
      {"lhs", lhs},
      {"rhs", rhs},
  };
}

bool TensorDimConstraint::operator==(TensorDimConstraint const &other) const {
  return type == other.type && lhs == other.lhs && rhs == other.rhs;
}

z3::expr TensorDimConstraint::to_z3(z3::context &c) const {
  z3::expr l = lhs.dim_expr->to_z3(c), r = rhs.dim_expr->to_z3(c);
  switch (type) {
    case ConstraintType::EQUAL:
      return l == r;
    case ConstraintType::EQUAL_OR_ONE:
      return l == r || l == 1 || r == 1;
    default:
      assert(false && "Unsupported constraint type");
  }
}

TensorDimConstraint make_equal_constraint(SymbolicTensorDim lhs,
                                          SymbolicTensorDim rhs) {
  return TensorDimConstraint(ConstraintType::EQUAL, lhs, rhs);
}

TensorDimConstraint make_equal_or_one_constraint(SymbolicTensorDim lhs,
                                                 SymbolicTensorDim rhs) {
  return TensorDimConstraint(ConstraintType::EQUAL_OR_ONE, lhs, rhs);
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
    std::shared_ptr<TensorDimExpr const> el = constraint.lhs.dim_expr,
                                         er = constraint.rhs.dim_expr;
    if (constraint.type == ConstraintType::EQUAL) {
      return probably_equal(el, er);
    }
    if (constraint.type == ConstraintType::EQUAL_OR_ONE) {
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
  hash_combine(seed, constraint.lhs);
  hash_combine(seed, constraint.rhs);
  return seed;
}

} // namespace std