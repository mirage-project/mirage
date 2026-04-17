#include "mirage/search/symbolic_graph/symbolic_tensor.h"
#include "mirage/search/symbolic_graph/tensor_dim_expr.h"

namespace mirage {
namespace search {

SymbolicDTensor::SymbolicDTensor(std::vector<SymbolicTensorDim> dim_templates)
    : dims(dim_templates) {}

SymbolicSTensor::SymbolicSTensor(std::vector<SymbolicTensorDim> dim_templates,
                                 bool after_accum)
    : dims(dim_templates), after_accum(after_accum) {}

SymbolicDTensor::operator json() const {
  std::vector<json> dims_json;
  for (auto const &dim : dims) {
    dims_json.push_back(json(*dim));
  }
  return json{{"dims", dims_json}};
}

SymbolicSTensor::operator json() const {
  std::vector<json> dims_json;
  for (auto const &dim : dims) {
    dims_json.push_back(json(*dim));
  }
  return json{{"dims", dims_json}, {"after_accum", after_accum}};
}

void from_json(json const &j, SymbolicDTensor &tensor) {
  std::vector<SymbolicTensorDim> dims;
  for (auto const &jdim : j.at("dims")) {
    SymbolicTensorDim dim;
    from_json(jdim, dim);
    dims.push_back(dim);
  }
  tensor = SymbolicDTensor(dims);
}

void from_json(json const &j, SymbolicSTensor &tensor) {
  std::vector<SymbolicTensorDim> dims;
  for (auto const &jdim : j.at("dims")) {
    SymbolicTensorDim dim;
    from_json(jdim, dim);
    dims.push_back(dim);
  }
  bool after_accum = j.at("after_accum").get<bool>();
  tensor = SymbolicSTensor(dims, after_accum);
}

} // namespace search
} // namespace mirage
