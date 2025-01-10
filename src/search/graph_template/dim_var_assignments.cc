#include "mirage/search/graph_template/dim_var_assignments.h"
#include "mirage/search/graph_template/tensor_dim_template.h"

namespace mirage {
namespace search {

void DimVarAssignments::assign(tensor_dim_var_index_t dim_var_index, int value) {
  assignments[dim_var_index] = value;
}

int DimVarAssignments::get_value(TensorDimTemplate const &dim_template) const {
  return dim_template.dim_expr->get_value(*this);
}

int DimVarAssignments::get_value(tensor_dim_var_index_t dim_var_index) const {
  return assignments.at(dim_var_index);
}

}
}
