#pragma once

#include "mirage/search/symbolic_graph/symbolic_tensor_dim.h"
#include "mirage/utils/json_utils.h"

#include <vector>

namespace mirage {
namespace search {

enum class ConstraintType {
  EQUAL,
  EQUAL_OR_ONE,
  GEQ,
  LEQ,
};

NLOHMANN_JSON_SERIALIZE_ENUM(ConstraintType,
                             {
                                 {ConstraintType::EQUAL, "EQUAL"},
                                 {ConstraintType::EQUAL_OR_ONE, "EQUAL_OR_ONE"},
                                 {ConstraintType::GEQ, "GEQ"},
                                 {ConstraintType::LEQ, "LEQ"},
                             })

class TensorDimConstraint {
public:
  TensorDimConstraint(ConstraintType type, std::vector<SymbolicTensorDim> dims);
  ConstraintType type;
  std::vector<SymbolicTensorDim> dims;

  operator json() const;
  bool operator==(TensorDimConstraint const &other) const;
  z3::expr to_z3(z3::context &c, DimVarAssignments const &assign) const;
};

TensorDimConstraint make_equal_constraint(SymbolicTensorDim lhs,
                                          SymbolicTensorDim rhs);

TensorDimConstraint make_geq_constraint(SymbolicTensorDim lhs,
                                        SymbolicTensorDim rhs);

TensorDimConstraint make_leq_constraint(SymbolicTensorDim lhs,
                                        SymbolicTensorDim rhs);

TensorDimConstraint make_equal_or_one_constraint(SymbolicTensorDim lhs,
                                                 SymbolicTensorDim rhs);

TensorDimConstraint make_non_negative_constraint(SymbolicTensorDim dim);

TensorDimConstraint make_non_positive_constraint(SymbolicTensorDim dim);

TensorDimConstraint
    make_sum_leq_one_constraint(std::vector<SymbolicTensorDim> dims);

TensorDimConstraint
    make_sum_geq_zero_constraint(std::vector<SymbolicTensorDim> dims);

bool check_satisfiability(
    std::unordered_set<TensorDimConstraint> const &pre_conds,
    std::unordered_set<TensorDimConstraint> const &constraints);

} // namespace search
} // namespace mirage

namespace std {

template <>
struct hash<mirage::search::TensorDimConstraint> {
  size_t
      operator()(mirage::search::TensorDimConstraint const &constraint) const;
};

} // namespace std
