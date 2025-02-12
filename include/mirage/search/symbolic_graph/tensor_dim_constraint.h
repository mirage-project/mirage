#pragma once

#include "mirage/search/symbolic_graph/symbolic_tensor_dim.h"
#include "mirage/utils/json_utils.h"

#include <vector>

namespace mirage {
namespace search {

enum class ConstraintType {
  EQUAL,
  EQUAL_OR_ONE,
};

NLOHMANN_JSON_SERIALIZE_ENUM(ConstraintType,
                             {
                                 {ConstraintType::EQUAL, "EQUAL"},
                                 {ConstraintType::EQUAL_OR_ONE, "EQUAL_OR_ONE"},
                             })

class TensorDimConstraint {
public:
  TensorDimConstraint(ConstraintType type,
                      SymbolicTensorDim lhs,
                      SymbolicTensorDim rhs);
  ConstraintType type;
  SymbolicTensorDim lhs, rhs;

  operator json() const;
  bool operator==(TensorDimConstraint const &other) const;
  z3::expr to_z3(z3::context &c) const;
};

TensorDimConstraint make_equal_constraint(SymbolicTensorDim lhs,
                                          SymbolicTensorDim rhs);

TensorDimConstraint make_equal_or_one_constraint(SymbolicTensorDim lhs,
                                                 SymbolicTensorDim rhs);

bool check_satisfiability(std::unordered_set<TensorDimConstraint> const &pre_conds,
                          std::unordered_set<TensorDimConstraint> const &constraints);

} // namespace search
} // namespace mirage

namespace std {

template <>
struct hash<mirage::search::TensorDimConstraint> {
  size_t operator()(mirage::search::TensorDimConstraint const &constraint) const;
};

} // namespace std
