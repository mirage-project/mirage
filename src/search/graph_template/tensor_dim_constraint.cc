#include "mirage/search/graph_template/tensor_dim_constraint.h"

#include <unordered_map>

namespace mirage {
namespace search {

TensorDimConstraint::TensorDimConstraint(ConstraintType type, SymbolicTensorDim lhs, SymbolicTensorDim rhs)
    : type(type), lhs(lhs), rhs(rhs) {}

TensorDimConstraint make_equal_constraint(SymbolicTensorDim lhs, SymbolicTensorDim rhs) {
  return TensorDimConstraint(ConstraintType::EQUAL, lhs, rhs);
}

bool check_satisfiability(std::vector<TensorDimConstraint> const &pre_conds,
                          std::vector<TensorDimConstraint> const &constraints) {
  auto probably_equal = [](std::shared_ptr<TensorDimExpr> el, std::shared_ptr<TensorDimExpr> er) {
    {
      TensorDimConst *cl = dynamic_cast<TensorDimConst *>(el.get());
      TensorDimConst *cr = dynamic_cast<TensorDimConst *>(er.get());
      if (cl && cr) {
        if (cl->value != cr->value) {
          return false;
        }
      }
    }
    {
      TensorDimDiv *dl = dynamic_cast<TensorDimDiv *>(el.get());
      TensorDimDiv *dr = dynamic_cast<TensorDimDiv *>(er.get());
      if (dl && dr) {
        TensorDimConst *cll = dynamic_cast<TensorDimConst *>(dl->lhs.get());
        TensorDimConst *crl = dynamic_cast<TensorDimConst *>(dr->lhs.get());
        TensorDimVar *clr = dynamic_cast<TensorDimVar *>(dl->rhs.get());
        TensorDimVar *crr = dynamic_cast<TensorDimVar *>(dr->rhs.get());
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
    std::shared_ptr<TensorDimExpr> el = constraint.lhs.dim_expr, er = constraint.rhs.dim_expr;
    if (constraint.type == ConstraintType::EQUAL) {
      return probably_equal(el, er);
    }
    if (constraint.type == ConstraintType::EQUAL_OR_ONE) {
      if (probably_equal(el, er)) {
        return true;
      }
      {
        TensorDimConst *cl = dynamic_cast<TensorDimConst *>(el.get());
        TensorDimConst *cr = dynamic_cast<TensorDimConst *>(er.get());
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

}
}
